import torch
import math
import torchaudio.compliance.kaldi as kaldi
import numpy as np
import sys
from pathlib import Path


def Fbank(wav):
    mat = kaldi.fbank(
        wav,
        num_mel_bins=80,
        frame_length=25,
        frame_shift=10,
        dither=0.0,
        sample_frequency=16000,
        window_type="hamming",
        use_energy=False,
    )
    # mat = mat - torch.mean(mat, dim=0)/(torch.std(mat,dim=0)+1e-10)
    mat = mat - torch.mean(mat, dim=0)
    return mat


similarity = torch.nn.CosineSimilarity(dim=-1, eps=1e-6)


def _sv_embedding_from_wav(model, wav_1xT, device="cuda", model_type="ecapa"):
    if model_type == "wavlm":
        raise NotImplementedError("FAKEBOB SV helper currently supports FBank-based backbones only.")
    feat = Fbank(wav_1xT.cpu()).unsqueeze(0).to(device)
    outputs = model(feat)
    return outputs[-1] if isinstance(outputs, tuple) else outputs


class FakeBobSV:
    """
    Query-based FAKEBOB-style black-box attack for speaker verification.

    This implementation follows the core recipe of FAKEBOB / SpeakerGuard:
    NES gradient estimation, momentum update, plateau-based learning-rate decay,
    and margin-based decision loss on top of a black-box SV score.
    """

    def __init__(
        self,
        model,
        enroll_emb,
        threshold=0.0,
        target_decision="accept",
        confidence=0.0,
        epsilon=0.002,
        max_iter=1000,
        max_lr=0.001,
        min_lr=1e-6,
        samples_per_draw=50,
        sigma=0.001,
        momentum=0.9,
        plateau_length=5,
        plateau_drop=2.0,
        eot_steps=1,
        query_batch_size=32,
        device="cuda",
        model_type="ecapa",
    ):
        self.model = model
        self.enroll_emb = enroll_emb.detach().to(device)
        self.threshold = float(threshold)
        self.target_decision = target_decision
        self.confidence = float(confidence)
        self.epsilon = float(epsilon)
        self.max_iter = int(max_iter)
        self.max_lr = float(max_lr)
        self.min_lr = float(min_lr)
        self.samples_per_draw = int(samples_per_draw)
        self.sigma = float(sigma)
        self.momentum = float(momentum)
        self.plateau_length = int(plateau_length)
        self.plateau_drop = float(plateau_drop)
        self.eot_steps = int(eot_steps)
        self.query_batch_size = int(query_batch_size)
        self.device = device
        self.model_type = model_type

        if self.samples_per_draw % 2 != 0:
            raise ValueError("samples_per_draw should be even for antithetic NES sampling.")
        if self.target_decision not in {"accept", "reject"}:
            raise ValueError("target_decision should be either 'accept' or 'reject'.")

    def _score_batch(self, wav_batch):
        """
        wav_batch: [B, 1, T] in normalized waveform range [-1, 1].
        returns: [B] cosine similarity scores.
        """
        wav_batch = wav_batch.to(self.device)
        scores = []
        scale = float(1 << 15)

        with torch.no_grad():
            for start in range(0, wav_batch.shape[0], self.query_batch_size):
                chunk = wav_batch[start : start + self.query_batch_size]
                for wav in chunk:
                    wav_1xT = wav
                    if wav_1xT.dim() == 1:
                        wav_1xT = wav_1xT.unsqueeze(0)
                    query = wav_1xT * scale
                    score_sum = 0.0
                    for _ in range(self.eot_steps):
                        emb = _sv_embedding_from_wav(self.model, query, device=self.device, model_type=self.model_type)
                        score_sum = score_sum + similarity(emb, self.enroll_emb).view(())
                    scores.append(score_sum / float(self.eot_steps))

        return torch.stack(scores)

    def _loss_from_scores(self, scores):
        if self.target_decision == "accept":
            return self.threshold + self.confidence - scores
        return scores + self.confidence - self.threshold

    def _estimate_grad(self, audio_1xT):
        """
        audio_1xT: [1, T] normalized waveform in [-1, 1]
        """
        n = audio_1xT.shape[-1]
        half = self.samples_per_draw // 2
        noise = torch.randn((half, 1, n), device=self.device)
        noise = torch.cat((noise, -noise), dim=0)
        zeros = torch.zeros((1, 1, n), device=self.device)
        noise_full = torch.cat((zeros, noise), dim=0)
        eval_batch = torch.clamp(audio_1xT.unsqueeze(0) + self.sigma * noise_full, -1.0, 1.0)

        scores = self._score_batch(eval_batch)
        losses = self._loss_from_scores(scores)
        adv_loss = losses[0]
        adv_score = scores[0]
        mean_loss = losses[1:].mean()
        grad = torch.mean(losses[1:].view(-1, 1, 1) * noise, dim=0) / max(self.sigma, 1e-12)
        return mean_loss, grad, adv_loss, adv_score

    def attack(self, audio):
        """
        audio: [T] or [1, T], normalized waveform in [-1, 1]
        """
        if audio.dim() == 1:
            audio = audio.unsqueeze(0)
        audio = audio.to(self.device)

        adv = audio.clone()
        best_adv = adv.clone()
        best_loss = float("inf")
        best_score = None
        grad = torch.zeros_like(adv)
        lr = self.max_lr
        last_losses = []
        lower = torch.clamp(audio - self.epsilon, min=-1.0, max=1.0)
        upper = torch.clamp(audio + self.epsilon, min=-1.0, max=1.0)
        history = []

        for iteration in range(self.max_iter):
            prev_grad = grad.clone()
            mean_loss, grad_est, adv_loss, adv_score = self._estimate_grad(adv)
            loss_value = float(adv_loss.item())
            score_value = float(adv_score.item())
            history.append(
                {
                    "iter": iteration,
                    "loss": loss_value,
                    "score": score_value,
                    "lr": lr,
                }
            )

            if loss_value < best_loss:
                best_loss = loss_value
                best_adv = adv.clone()
                best_score = score_value

            if loss_value < 0.0:
                return best_adv, True, best_score, history

            grad = self.momentum * prev_grad + (1.0 - self.momentum) * grad_est
            last_losses.append(float(mean_loss.item()))
            last_losses = last_losses[-self.plateau_length :]
            if len(last_losses) == self.plateau_length and last_losses[-1] > last_losses[0]:
                if lr > self.min_lr:
                    lr = max(lr / self.plateau_drop, self.min_lr)
                last_losses = []

            adv = adv - lr * torch.sign(grad)
            adv = torch.min(torch.max(adv, lower), upper).detach()

        return best_adv, False, best_score, history


def fakebob_sv(
    model,
    wav1,
    emb_2,
    threshold=0.0,
    target_decision="accept",
    confidence=0.0,
    epsilon=30.0,
    max_iter=1000,
    max_lr=1.0,
    min_lr=1e-3,
    samples_per_draw=50,
    sigma=1.0,
    momentum=0.9,
    plateau_length=5,
    plateau_drop=2.0,
    eot_steps=1,
    query_batch_size=32,
    device="cuda",
    model_type="ecapa",
):
    """
    Convenience wrapper that keeps compatibility with the current SV attack code.

    wav1 may be either in normalized [-1, 1] waveform scale or in the current
    int16-style attack scale [-32768, 32767]. epsilon/max_lr/sigma follow the
    same convention and are converted automatically when needed.
    """
    scale = float(1 << 15)
    if wav1.dim() == 1:
        wav1 = wav1.unsqueeze(0)
    wav1 = wav1.detach().clone().to(device)

    use_int16_scale = float(torch.max(torch.abs(wav1)).item()) > 2.0
    if use_int16_scale:
        wav_norm = wav1 / scale
        epsilon = epsilon / scale
        max_lr = max_lr / scale
        min_lr = min_lr / scale
        sigma = sigma / scale
    else:
        wav_norm = wav1

    attacker = FakeBobSV(
        model=model,
        enroll_emb=emb_2,
        threshold=threshold,
        target_decision=target_decision,
        confidence=confidence,
        epsilon=epsilon,
        max_iter=max_iter,
        max_lr=max_lr,
        min_lr=min_lr,
        samples_per_draw=samples_per_draw,
        sigma=sigma,
        momentum=momentum,
        plateau_length=plateau_length,
        plateau_drop=plateau_drop,
        eot_steps=eot_steps,
        query_batch_size=query_batch_size,
        device=device,
        model_type=model_type,
    )

    adv_norm, success, best_score, history = attacker.attack(wav_norm)

    noise = wav_norm - adv_norm
    P = lambda x: torch.sum(torch.pow(x, 2))
    snr = round(10 * math.log10(float(P(wav_norm) / (P(noise) + 1e-12))), 2)

    if use_int16_scale:
        adv_return = (adv_norm * scale).detach().cpu()
    else:
        adv_return = adv_norm.detach().cpu()

    score_tensor = torch.tensor(0.0 if best_score is None else best_score)
    return adv_return, score_tensor, snr, success, history

_PNP_RUNTIME = None
_PNP_MODEL_CACHE = {}


def _load_pnp_runtime():
    global _PNP_RUNTIME
    if _PNP_RUNTIME is not None:
        return _PNP_RUNTIME

    pnp_root = Path(__file__).resolve().parents[1] / "pnp_asv" / "pnp"
    pnp_root_str = str(pnp_root)
    if pnp_root_str not in sys.path:
        sys.path.insert(0, pnp_root_str)

    from params_pnp import AttrDict, params as base_params_pnp
    from model_pnp import DiffWave as PnpDiffWave

    _PNP_RUNTIME = {
        "AttrDict": AttrDict,
        "params": base_params_pnp,
        "Model": PnpDiffWave,
    }
    return _PNP_RUNTIME


def load_pnp_model(ckpt_path, device="cuda"):
    cache_key = (str(ckpt_path), str(device))
    if cache_key in _PNP_MODEL_CACHE:
        return _PNP_MODEL_CACHE[cache_key]

    runtime = _load_pnp_runtime()
    checkpoint = torch.load(ckpt_path, map_location=device)
    model = runtime["Model"](runtime["AttrDict"](runtime["params"])).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    for param in model.parameters():
        param.requires_grad_(False)
    _PNP_MODEL_CACHE[cache_key] = model
    return model


def pnp_forward_grad(
    pnp_model,
    audio_1xT,
    t_index,
    params=None,
    lam=0.7,
    simple_add=False,
    noise=None,
    clamp=True,
):
    """
    Differentiable PnP forward pass for adaptive attacks.

    This mirrors src/pnp/inference_pnp.py but intentionally keeps autograd enabled.
    audio_1xT should be shaped [1, T] and be in the waveform domain [-1, 1].
    """
    assert audio_1xT.ndim == 2 and audio_1xT.size(0) == 1, "audio should be [1, T]"

    runtime = _load_pnp_runtime()
    if params is None:
        params = runtime["params"]

    device = audio_1xT.device
    dtype = audio_1xT.dtype

    training_noise_schedule = np.array(params.noise_schedule)
    alpha = 1.0 - training_noise_schedule
    alpha_bar = np.cumprod(alpha)
    t_idx = int(np.clip(t_index, 1, len(alpha_bar))) - 1

    sqrt_alpha_bar_t = torch.tensor(alpha_bar[t_idx] ** 0.5, device=device, dtype=dtype)
    sqrt_one_minus_alpha_bar_t = torch.tensor((1.0 - alpha_bar[t_idx]) ** 0.5, device=device, dtype=dtype)

    if simple_add:
        t_tensor = torch.zeros(1, device=device, dtype=torch.float32)
    else:
        # Keep the same 1-based float convention as the current inference code.
        t_tensor = torch.tensor([t_idx + 1], device=device, dtype=torch.float32)

    eps_dir = pnp_model(audio_1xT, t_tensor)
    if eps_dir.ndim == 3:
        eps_dir = eps_dir.squeeze(1)
    eps_dir = eps_dir / (eps_dir.norm(p=2, dim=1, keepdim=True) + 1e-8)

    if noise is None:
        noise = torch.randn_like(audio_1xT)
    eps_tilde = lam * eps_dir + (1.0 - lam ** 2) ** 0.5 * noise

    if simple_add:
        out = audio_1xT + eps_tilde
    else:
        out = sqrt_alpha_bar_t * audio_1xT + sqrt_one_minus_alpha_bar_t * eps_tilde

    if clamp:
        out = torch.clamp(out, -1.0, 1.0)
    return out


def pgd_2_adaptive_pnp(
    model,
    pnp_model,
    wav1,
    emb_2,
    labels,
    t_index=1,
    simple_add=False,
    lam=0.7,
    eot_steps=4,
    purifier_clamp=True,
    loss=torch.nn.CosineEmbeddingLoss(margin=-1),
    eps=6400,
    eps_for_division=1e-10,
    alpha=1,
    iters=50,
    device="cuda",
    model_type="ecapa",
):
    """
    Adaptive PGD-L2 attack against the composed system ASV(PNP(x)).

    wav1 is kept in the same attack space as the existing PGD code, i.e. the waveform
    is typically scaled by 2^15 before FBank extraction.
    """
    if model_type == "wavlm":
        raise NotImplementedError("pgd_2_adaptive_pnp currently supports FBank-based ASV backbones only.")

    ori_wav = wav1.detach().to(device)
    wav1 = wav1.detach().to(device)
    scale = float(1 << 15)

    for _ in range(iters):
        grad = torch.zeros_like(wav1, device=device)
        for _eot in range(eot_steps):
            wav1_eot = wav1.detach().clone().requires_grad_(True)
            model.zero_grad()
            purified = pnp_forward_grad(
                pnp_model,
                wav1_eot / scale,
                t_index=t_index,
                lam=lam,
                simple_add=simple_add,
                clamp=purifier_clamp,
            )
            feat1 = Fbank((purified * scale).cpu()).unsqueeze(0).to(device)
            outputs = model(feat1)
            emb_1 = outputs[-1] if isinstance(outputs, tuple) else outputs
            cost = loss(emb_1, emb_2, labels).to(device)
            grad_step = torch.autograd.grad(cost, wav1_eot, retain_graph=False, create_graph=False)[0]
            grad = grad + grad_step.detach()
            del wav1_eot, purified, feat1, outputs, emb_1, cost, grad_step
        grad = grad / float(eot_steps)
        grad_norms = torch.norm(grad.view(1, -1), p=2, dim=1) + eps_for_division
        grad = grad / grad_norms
        adv_wav = wav1.detach() + alpha * grad
        eta = adv_wav - ori_wav
        eta_norms = torch.norm(eta.view(1, -1), p=2, dim=1)
        factor = eps / eta_norms
        factor = torch.min(factor, torch.ones_like(eta_norms))
        eta = eta * factor.view(-1, 1)
        wav1 = torch.clamp(ori_wav + eta, min=-32768, max=32767).detach()

    noise = torch.sub(ori_wav, wav1)
    P = lambda x: torch.sum(torch.pow(x, 2))
    SNR = round(10 * math.log10(P(ori_wav) / P(noise)), 2)

    with torch.no_grad():
        sims = []
        for _eot in range(eot_steps):
            purified = pnp_forward_grad(
                pnp_model,
                wav1 / scale,
                t_index=t_index,
                lam=lam,
                simple_add=simple_add,
                clamp=purifier_clamp,
            )
            feat1 = Fbank((purified * scale).cpu()).unsqueeze(0).to(device)
            outputs = model(feat1)
            emb_1 = outputs[-1] if isinstance(outputs, tuple) else outputs
            sims.append(similarity(emb_2, emb_1))
        sim = torch.stack(sims).mean(dim=0)

    wav1 = (wav1 / scale).cpu()
    return wav1, sim, SNR


def pgd_i_adaptive_pnp(
    model,
    pnp_model,
    wav1,
    emb_2,
    labels,
    t_index=1,
    simple_add=False,
    lam=0.7,
    eot_steps=4,
    purifier_clamp=True,
    loss=torch.nn.CosineEmbeddingLoss(margin=-1),
    eps=30,
    alpha=1,
    iters=50,
    device="cuda",
    model_type="ecapa",
):
    """
    Adaptive PGD-Linf attack against the composed system ASV(PNP(x)).
    """
    if model_type == "wavlm":
        raise NotImplementedError("pgd_i_adaptive_pnp currently supports FBank-based ASV backbones only.")

    ori_wav = wav1.detach().to(device)
    wav1 = wav1.detach().to(device)
    scale = float(1 << 15)

    for _ in range(iters):
        grad = torch.zeros_like(wav1, device=device)
        for _eot in range(eot_steps):
            wav1_eot = wav1.detach().clone().requires_grad_(True)
            model.zero_grad()
            purified = pnp_forward_grad(
                pnp_model,
                wav1_eot / scale,
                t_index=t_index,
                lam=lam,
                simple_add=simple_add,
                clamp=purifier_clamp,
            )
            feat1 = Fbank((purified * scale).cpu()).unsqueeze(0).to(device)
            outputs = model(feat1)
            emb_1 = outputs[-1] if isinstance(outputs, tuple) else outputs
            cost = loss(emb_1, emb_2, labels).to(device)
            grad_step = torch.autograd.grad(cost, wav1_eot, retain_graph=False, create_graph=False)[0]
            grad = grad + grad_step.detach()
            del wav1_eot, purified, feat1, outputs, emb_1, cost, grad_step
        grad = grad / float(eot_steps)
        adv_wav = wav1.detach() + alpha * grad.sign()
        eta = torch.clamp(adv_wav - ori_wav, min=-eps, max=eps)
        wav1 = torch.clamp(ori_wav + eta, min=-32768, max=32767).detach()

    noise = torch.sub(ori_wav, wav1)
    P = lambda x: torch.sum(torch.pow(x, 2))
    SNR = round(10 * math.log10(P(ori_wav) / P(noise)), 2)

    with torch.no_grad():
        sims = []
        for _eot in range(eot_steps):
            purified = pnp_forward_grad(
                pnp_model,
                wav1 / scale,
                t_index=t_index,
                lam=lam,
                simple_add=simple_add,
                clamp=purifier_clamp,
            )
            feat1 = Fbank((purified * scale).cpu()).unsqueeze(0).to(device)
            outputs = model(feat1)
            emb_1 = outputs[-1] if isinstance(outputs, tuple) else outputs
            sims.append(similarity(emb_2, emb_1))
        sim = torch.stack(sims).mean(dim=0)

    wav1 = (wav1 / scale).cpu()
    return wav1, sim, SNR


def mifgsm_adaptive_pnp(
    model,
    pnp_model,
    wav1,
    emb_2,
    labels,
    t_index=1,
    simple_add=False,
    lam=0.7,
    eot_steps=4,
    purifier_clamp=True,
    loss=torch.nn.CosineEmbeddingLoss(margin=-1),
    eps=30,
    alpha=1,
    decay=1.0,
    iters=50,
    device="cuda",
    model_type="ecapa",
):
    """
    Adaptive MI-FGSM attack against the composed system ASV(PNP(x)).
    """
    if model_type == "wavlm":
        raise NotImplementedError("mifgsm_adaptive_pnp currently supports FBank-based ASV backbones only.")

    ori_wav = wav1.detach().to(device)
    wav1 = wav1.detach().to(device)
    scale = float(1 << 15)
    momentum = torch.zeros_like(wav1, device=device)

    for _ in range(iters):
        grad = torch.zeros_like(wav1, device=device)
        for _eot in range(eot_steps):
            wav1_eot = wav1.detach().clone().requires_grad_(True)
            model.zero_grad()
            purified = pnp_forward_grad(
                pnp_model,
                wav1_eot / scale,
                t_index=t_index,
                lam=lam,
                simple_add=simple_add,
                clamp=purifier_clamp,
            )
            feat1 = Fbank((purified * scale).cpu()).unsqueeze(0).to(device)
            outputs = model(feat1)
            emb_1 = outputs[-1] if isinstance(outputs, tuple) else outputs
            cost = loss(emb_1, emb_2, labels).to(device)
            grad_step = torch.autograd.grad(cost, wav1_eot, retain_graph=False, create_graph=False)[0]
            grad = grad + grad_step.detach()
            del wav1_eot, purified, feat1, outputs, emb_1, cost, grad_step
        grad = grad / float(eot_steps)
        grad = grad / (torch.mean(torch.abs(grad), dim=1, keepdim=True) + 1e-10)
        grad = grad + momentum * decay
        momentum = grad

        adv_wav = wav1.detach() + alpha * grad.sign()
        delta = torch.clamp(adv_wav - ori_wav, min=-eps, max=eps)
        wav1 = torch.clamp(ori_wav + delta, min=-32768, max=32767).detach()

    noise = torch.sub(ori_wav, wav1)
    P = lambda x: torch.sum(torch.pow(x, 2))
    SNR = round(10 * math.log10(P(ori_wav) / P(noise)), 2)

    with torch.no_grad():
        sims = []
        for _eot in range(eot_steps):
            purified = pnp_forward_grad(
                pnp_model,
                wav1 / scale,
                t_index=t_index,
                lam=lam,
                simple_add=simple_add,
                clamp=purifier_clamp,
            )
            feat1 = Fbank((purified * scale).cpu()).unsqueeze(0).to(device)
            outputs = model(feat1)
            emb_1 = outputs[-1] if isinstance(outputs, tuple) else outputs
            sims.append(similarity(emb_2, emb_1))
        sim = torch.stack(sims).mean(dim=0)

    wav1 = (wav1 / scale).cpu()
    return wav1, sim, SNR


def pgd_i(
    model,
    wav1,
    emb_2,
    labels,
    loss=torch.nn.CosineEmbeddingLoss(margin=-1),
    eps=30,
    alpha=1,
    iters=50,
    device="cuda",
    model_type="wavlm"
):
    """
    pgd l_infinite attack

    Arguments:
            model (nn.Module): model to attack.
            wav1: (torch.Tensor) input audio waveform,range:[-32768,32767].
            emb_2: (torch.Tensor) fbank feature of registered wav.
            labels: (torch.Tensor) ground truth labels, 1 for target, -1 for non-target.
            eps (float): maximum perturbation. (Default: 30)
            alpha (float): step size. (Default: 1)
            iters (int): number of iterations. (Default: 50)
            device (str): device to run the attack on. (Default: 'cuda')

    """
    ori_wav = wav1.data
    for _ in range(iters):
        wav1.requires_grad = True
        if model_type == "wavlm":
            outputs = model(wav1)
        else:
            feat1 = Fbank(wav1).unsqueeze(0).to(device)
            outputs = model(feat1)
        emb_1 = outputs[-1] if isinstance(outputs, tuple) else outputs
        model.zero_grad()
        cost = loss(emb_1, emb_2, labels).to(device)
        grad = torch.autograd.grad(cost, wav1, retain_graph=False, create_graph=False)[0]
        adv_wav = wav1.detach() + alpha * grad.sign()
        eta = torch.clamp(adv_wav - ori_wav, min=-eps, max=eps)
        wav1 = torch.clamp(ori_wav + eta, min=-32768, max=32767).detach()
    noise = torch.sub(ori_wav, wav1)
    P = lambda x: torch.sum(torch.pow(x, 2))
    SNR = round(10 * math.log10(P(ori_wav) / P(noise)), 2)
    if model_type == "wavlm":
            outputs = model(wav1)
    else:
        feat1 = Fbank(wav1).unsqueeze(0).to(device)
        outputs = model(feat1)
    emb_1 = outputs[-1] if isinstance(outputs, tuple) else outputs
    sim = similarity(emb_1, emb_2)
    if model_type=="ecapa" or model_type=="campp":
        wav1 = wav1 / (1 << 15)
    # print(f"loss:{cost.item()},cos sim:{sim.item()},SNR:{SNR}db")
    return wav1, sim, SNR


def pgd_2(
    model,
    wav1,
    emb_2,
    labels,
    loss=torch.nn.CosineEmbeddingLoss(margin=-1),
    eps=6400,
    eps_for_division=1e-10,
    alpha=1,
    iters=50,
    device="cuda",
    model_type="wavlm"
):
    """
    pgd l_2 attack

    Arguments:
            model (nn.Module): model to attack.
            wav1: (torch.Tensor) input audio waveform,range:[-32768,32767].
            emb_2: (torch.Tensor) fbank feature of registered wav.
            labels: (torch.Tensor) ground truth labels, 1 for target, -1 for non-target.
            eps (float): maximum perturbation. (Default: 6400)
            alpha (float): step size. (Default: 1)
            iters (int): number of iterations. (Default: 50)
            device (str): device to run the attack on. (Default: 'cuda')

    """
    ori_wav = wav1.data
    for _ in range(iters):
        wav1.requires_grad = True
        if model_type == "wavlm":
            outputs = model(wav1)
        else:
            feat1 = Fbank(wav1).unsqueeze(0).to(device)
            outputs = model(feat1)
        emb_1 = outputs[-1] if isinstance(outputs, tuple) else outputs
        model.zero_grad()
        cost = loss(emb_1, emb_2, labels).to(device)
        grad = torch.autograd.grad(cost, wav1, retain_graph=False, create_graph=False)[0]
        grad_norms = torch.norm(grad.view(1, -1), p=2, dim=1) + eps_for_division
        # print(grad.shape)
        grad = grad / grad_norms
        adv_wav = wav1.detach() + alpha * grad
        eta = adv_wav - ori_wav
        eta_norms = torch.norm(eta.view(1, -1), p=2, dim=1)
        factor = eps / eta_norms
        factor = torch.min(factor, torch.ones_like(eta_norms))
        eta = eta * factor.view(
            -1,
            1,
        )
        wav1 = torch.clamp(ori_wav + eta, min=-32768, max=32767).detach()
    noise = torch.sub(ori_wav, wav1)
    P = lambda x: torch.sum(torch.pow(x, 2))
    SNR = round(10 * math.log10(P(ori_wav) / P(noise)), 2)
    if model_type == "wavlm":
            outputs = model(wav1)
    else:
        feat1 = Fbank(wav1).unsqueeze(0).to(device)
        outputs = model(feat1)
    emb_1 = outputs[-1] if isinstance(outputs, tuple) else outputs
    sim = similarity(emb_2, emb_1)
    if model_type=="ecapa" or model_type=="campp":
        wav1 = wav1 / (1 << 15)
    # print(f"loss:{cost.item()},cos sim:{sim.item()},SNR:{SNR}db")
    return wav1, sim, SNR


def mifgsm(
    model,
    wav1,
    emb_2,
    labels,
    loss=torch.nn.CosineEmbeddingLoss(margin=-1),
    eps=30,
    alpha=1,
    decay=1.0,
    iters=50,
    device="cuda",
    model_type="wavlm"
):
    """
    mifgsm l_infinite attack

    Arguments:
            model (nn.Module): model to attack.
            wav1: (torch.Tensor) input audio waveform,range:[-32768,32767].
            emb_2: (torch.Tensor) fbank feature of registered wav.
            labels: (torch.Tensor) ground truth labels, 1 for target, -1 for non-target.
            eps (float): maximum perturbation. (Default: 30)
            alpha (float): step size. (Default: 1)
            decay (float): momentum decay factor. (Default: 1.0)
            iters (int): number of iterations. (Default: 50)
            device (str): device to run the attack on. (Default: 'cuda')

    """

    ori_wav = wav1.data
    momentum = torch.zeros_like(wav1)

    labels = labels.to(device)
    for i in range(iters):
        wav1.requires_grad = True
        feat1 = Fbank(wav1).unsqueeze(0).to(device)
        outputs = model(feat1)
        emb_1 = outputs[-1] if isinstance(outputs, tuple) else outputs
        model.zero_grad()
        cost = loss(emb_1, emb_2, labels).to(device)
        # print(cost)
        grad = torch.autograd.grad(cost, wav1, retain_graph=False, create_graph=False,allow_unused=True)[0]
        grad = grad / torch.mean(torch.abs(grad), dim=(1), keepdim=True)
        grad = grad + momentum * decay
        momentum = grad

        adv_wav = wav1.detach() + alpha * grad.sign()
        delta = torch.clamp(adv_wav - ori_wav, min=-eps, max=eps)
        wav1 = ori_wav + delta
        wav1 = torch.clamp(wav1, min=-32768, max=32767)

    noise = torch.sub(ori_wav, wav1)
    P = lambda x: torch.sum(torch.pow(x, 2))
    SNR = round(10 * math.log10(P(ori_wav) / P(noise)), 2)
    if model_type == "wavlm":
            outputs = model(wav1)
    else:
        feat1 = Fbank(wav1).unsqueeze(0).to(device)
        outputs = model(feat1)
    emb_1 = outputs[-1] if isinstance(outputs, tuple) else outputs
    sim = similarity(emb_2, emb_1)
    if model_type=="ecapa" or model_type=="campp":
        wav1 = wav1 / (1 << 15)

    return wav1, sim, SNR


def pgd_i_wavlm(
    model,
    wav1,
    wav2,
    labels,
    feature_extractor,
    loss=torch.nn.CosineEmbeddingLoss(margin=-1),
    eps=30,
    alpha=0.001,
    iters=50,
    device="cuda",):
    """
    pgd l_infinite attack for wavlm

    Arguments:
            model (nn.Module): model to attack.
            wav1: (torch.Tensor) input audio waveform to be attacked.
            wav2: (torch.Tensor) input audio waveform of enrollment wav.
            labels: (torch.Tensor) ground truth labels, 1 for target, -1 for non-target.
            feature_extractor: (nn.Module) feature extractor for wavlm.
            eps (float): maximum perturbation.
            alpha (float): step size.
            iters (int): number of iterations.
            device (str): device to run the attack on. (Default: 'cuda')

    """
    ori_wav = wav1.detach().to(device)
    wav1, wav2 = wav1[0].numpy(), wav2[0].numpy()
    wav1 = feature_extractor(
        [wav1], sampling_rate=16000, return_tensors="pt", padding=True
    ).to(device)
    wav2 = feature_extractor(
        [wav2], sampling_rate=16000, return_tensors="pt", padding=True
    ).to(device)
    emb_2 = model(**wav2).embeddings
    emb_2 = torch.nn.functional.normalize(emb_2, dim=-1)

    for _ in range(iters):
        wav1.input_values.requires_grad = True
        emb_1 = model(**wav1).embeddings
        emb_1 = torch.nn.functional.normalize(emb_1, dim=-1)
        cost = loss(emb_1, emb_2, labels).to(device)
        grad = torch.autograd.grad(
            cost,
            wav1.input_values,
            retain_graph=True,
            create_graph=False,
        )[0]
        adv_wav = wav1.input_values.detach() + alpha * grad.sign()
        eta = torch.clamp(adv_wav - ori_wav, min=-eps, max=eps)
        wav1 = torch.clamp(ori_wav + eta, min=-1, max=1).detach()
        wav1 = feature_extractor(
            wav1.cpu().numpy(), sampling_rate=16000, return_tensors="pt", padding=True
        ).to(device)
    noise = torch.sub(ori_wav, wav1["input_values"])
    P = lambda x: torch.sum(torch.pow(x, 2))
    SNR = round(10 * math.log10(P(ori_wav) / P(noise)), 2)
    with torch.no_grad():
        emb_1 = model(**wav1).embeddings

    sim = similarity(emb_1, emb_2)
    # print(f"loss:{cost.item()},cos sim:{sim.item()},SNR:{SNR}db")
    return wav1["input_values"], sim, SNR


def pgd_2_wavlm(
    model,
    wav1,
    wav2,
    labels,
    feature_extractor,
    loss=torch.nn.CosineEmbeddingLoss(margin=-1),
    eps=0.3,
    eps_for_division=1e-10,
    alpha=0.001,
    iters=50,
    device="cuda",):
    """
    pgd l_2 attack for wavlm

    Arguments:
            model (nn.Module): model to attack.
            wav1: (torch.Tensor) input audio waveform to be attacked.
            wav2: (torch.Tensor) input audio waveform of enrollment wav.
            labels: (torch.Tensor) ground truth labels, 1 for target, -1 for non-target.
            feature_extractor: (nn.Module) feature extractor for wavlm.
            eps (float): maximum perturbation.
            alpha (float): step size.
            iters (int): number of iterations.
            device (str): device to run the attack on. (Default: 'cuda')

    """
    ori_wav = wav1.detach().to(device)
    wav1, wav2 = wav1[0].numpy(), wav2[0].numpy()
    wav1 = feature_extractor(
        [wav1], sampling_rate=16000, return_tensors="pt", padding=True
    ).to(device)
    wav2 = feature_extractor(
        [wav2], sampling_rate=16000, return_tensors="pt", padding=True
    ).to(device)
    emb_2 = model(**wav2).embeddings
    emb_2 = torch.nn.functional.normalize(emb_2, dim=-1)
    for _ in range(iters):
        wav1.input_values.requires_grad = True
        emb_1 = model(**wav1).embeddings
        emb_1 = torch.nn.functional.normalize(emb_1, dim=-1)
        cost = loss(emb_1, emb_2, labels).to(device)
        grad = torch.autograd.grad(
            cost,
            wav1.input_values,
            retain_graph=True,
            create_graph=False,
        )[0]
        grad_norms = torch.norm(grad.view(1, -1), p=2, dim=1) + eps_for_division
        grad = grad / grad_norms
        adv_wav = wav1.input_values.detach() + alpha * grad.sign()
        eta = adv_wav - ori_wav
        eta_norms = torch.norm(eta.view(1, -1), p=2, dim=1)
        factor = eps / eta_norms
        factor = torch.min(factor, torch.ones_like(eta_norms))
        eta = eta * factor.view(
            -1,
            1,
        )
        wav1 = torch.clamp(ori_wav + eta, min=-1, max=1)
        wav1 = feature_extractor(
            wav1.cpu().numpy(), sampling_rate=16000, return_tensors="pt", padding=True
        ).to(device)
    noise = torch.sub(ori_wav, wav1["input_values"])
    P = lambda x: torch.sum(torch.pow(x, 2))
    SNR = round(10 * math.log10(P(ori_wav) / P(noise)), 2)
    with torch.no_grad():
        emb_1 = model(**wav1).embeddings

    sim = similarity(emb_1, emb_2)
    # print(f"loss:{cost.item()},cos sim:{sim.item()},SNR:{SNR}db")
    return wav1["input_values"], sim, SNR

def mifgsm_wavlm(
    model,
    wav1,
    wav2,
    labels,
    feature_extractor,
    loss=torch.nn.CosineEmbeddingLoss(margin=-1),
    eps=0.03,
    alpha=0.001,
    decay=1.0,
    iters=50,
    device="cuda",
):
    """
    MI-FGSM (L_infinity) attack for WavLM.

    Arguments:
        model (nn.Module): WavLM model (returning .embeddings).
        wav1 (torch.Tensor): test waveform to attack, shape (1, T) or (T,), float in [-1, 1].
        wav2 (torch.Tensor): enrollment waveform, shape (1, T) or (T,), float in [-1, 1].
        labels (torch.Tensor): ground truth labels, 1 for target, -1 for non-target.
        feature_extractor: HF feature extractor for WavLM.
        loss: cosine embedding loss (default margin=-1 for targeted setting in your code).
        eps (float): max L_inf perturbation on waveform scale (default 0.03).
        alpha (float): step size (default 0.001).
        decay (float): momentum decay factor.
        iters (int): number of iterations.
        device (str): device.

    Returns:
        adv_wav (torch.Tensor): adversarial waveform tensor, shape (1, T).
        sim (torch.Tensor or float): cosine similarity between emb_1 and emb_2 (depends on your similarity()).
        SNR (float): signal-to-noise ratio in dB between ori and adv.
    """
    model.eval()
    ori_wav = wav1.detach().to(device)  
    wav1_np = wav1[0].detach().cpu().numpy()
    wav2_np = wav2[0].detach().cpu().numpy()

    wav1_inputs = feature_extractor(
        [wav1_np], sampling_rate=16000, return_tensors="pt", padding=True
    ).to(device)
    wav2_inputs = feature_extractor(
        [wav2_np], sampling_rate=16000, return_tensors="pt", padding=True
    ).to(device)

    labels = labels.to(device)

    with torch.no_grad():
        emb_2 = model(**wav2_inputs).embeddings
        emb_2 = torch.nn.functional.normalize(emb_2, dim=-1)

    momentum = torch.zeros_like(ori_wav).to(device)

    for _ in range(iters):
        wav1_inputs.input_values.requires_grad = True

        emb_1 = model(**wav1_inputs).embeddings
        emb_1 = torch.nn.functional.normalize(emb_1, dim=-1)

        cost = loss(emb_1, emb_2, labels).to(device)
        grad = torch.autograd.grad(
            cost,
            wav1_inputs["input_values"],
            retain_graph=False,
            create_graph=False,
        )[0]  # (1, T)

        grad = grad / (torch.mean(torch.abs(grad), dim=-1, keepdim=True))

        grad = grad + decay * momentum
        momentum = grad.detach()

        adv = wav1_inputs["input_values"].detach() + alpha * grad.sign()

        delta = torch.clamp(adv - ori_wav, min=-eps, max=eps)
        # print(delta,eps)
        adv = torch.clamp(ori_wav + delta, min=-1.0, max=1.0).detach()

        wav1_inputs = feature_extractor(
            adv.detach().cpu().numpy(),
            sampling_rate=16000,
            return_tensors="pt",
            padding=True,
        ).to(device)

    adv_wav = wav1_inputs["input_values"].detach()

    noise = adv_wav - ori_wav
    P = lambda x: torch.sum(x ** 2)
    snr_val = 10.0 * torch.log10(P(ori_wav) / (P(noise) + 1e-10))
    SNR = round(snr_val.item(), 2)

    with torch.no_grad():
        emb_1 = model(**wav1_inputs).embeddings
        emb_1 = torch.nn.functional.normalize(emb_1, dim=-1)

    sim = similarity(emb_1, emb_2)
    return adv_wav, sim, SNR

def _to_attack_space(x, bounds=[-32768, 32767]):
    min_, max_ = bounds
    a = (min_ + max_) / 2
    b = (max_ - min_) / 2
    x = (x - a) / b  # map from [min_, max_] to [-1, +1]
    x = x * 0.999999  # from [-1, +1] to approx. (-1, +1)
    x = x.arctanh()  # from (-1, +1) to (-inf, +inf)
    return x


def _to_model_space(x, bounds=[-32768, 32767]):
    min_, max_ = bounds
    x = x.tanh()  # from (-inf, +inf) to (-1, +1)
    a = (min_ + max_) / 2
    b = (max_ - min_) / 2
    x = x * b + a  # map from (-1, +1) to (min_, max_)
    return x


def tanh_space(x):
    return 1 / 2 * (torch.tanh(x) + 1)


def atanh(x):
    return 0.5 * torch.log((1 + x) / (1 - x))


def inverse_tanh_space(x):
    # torch.atanh is only for torch >= 1.7.0
    # atanh is defined in the range -1 to 1
    return atanh(torch.clamp(x * 2 - 1, min=-32768, max=32767))


# f-function
def f(outputs, labels, device, kappa):
    # find the max logit other than the target class
    if labels == 0:
        other = 1.5 - outputs / 2
    else:
        other = 1.5 + outputs / 2

    print(other)

    return torch.clamp((other - outputs), min=-kappa)


def cw_attack(
    model,
    wav2,
    emb_t,
    labels,
    c=1,
    kappa=1,
    iters=50,
    lr=0.01,
    device="cuda",
):
    ori_wav = wav2.data.to(device)
    wav2 = _to_attack_space(wav2).detach().to(device)
    wav2.requires_grad = True
    best_adv_wav = wav2.clone().detach()
    best_L2 = 1e10 * torch.ones((len(wav2))).to(device)
    prev_cost = 1e10
    dim = len(wav2.shape)
    target_labels = (
        torch.tensor([0]).to(device) if labels[0] == 1 else torch.tensor([1]).to(device)
    )
    MSELoss = torch.nn.MSELoss(reduction="none")
    Flatten = torch.nn.Flatten()
    optimizer = torch.optim.Adam([wav2], lr=lr)

    for _ in range(iters):
        adv_wav = _to_model_space(wav2).to(device)
        print(adv_wav)
        current_L2 = MSELoss(adv_wav, ori_wav).sum(dim=-1).to(device)
        L2_loss = current_L2.sum().to(device)
        feat2 = Fbank(wav2).unsqueeze(0).to(device)
        outputs = model(feat2)
        emb_s = outputs[-1] if isinstance(outputs, tuple) else outputs
        model.zero_grad()
        f_loss = (
            f(similarity(emb_s, emb_t), labels.to(device), device, kappa)
            .sum()
            .to(device)
        )
        cost = f_loss + c * L2_loss
        optimizer.zero_grad()
        cost.backward(retain_graph=True)
        optimizer.step()
        pre = torch.argmax(outputs.detach(), 1)
        condition = (pre == target_labels).float().to(device)
        mask = condition * (best_L2 > current_L2.detach())
        best_L2 = mask * current_L2.detach() + (1 - mask) * best_L2
        mask = mask.view([-1] + [1] * (dim - 1))
        best_adv_wav = mask * adv_wav.detach() + (1 - mask) * best_adv_wav
        if _ % max(iters // 10, 1) == 0:
            if cost.item() > prev_cost:
                break
            prev_cost = cost.item()
    noise = torch.sub(ori_wav, best_adv_wav)
    P = lambda x: torch.sum(torch.pow(x, 2))
    SNR = round(10 * math.log10(P(ori_wav) / P(noise)), 2)
    feat2 = Fbank(wav2).unsqueeze(0).to(device)
    outputs = model(feat2)
    emb_s = outputs[-1] if isinstance(outputs, tuple) else outputs
    sim = similarity(emb_t, emb_s)
    # print(f"loss:{cost.item()},cos sim:{sim.item()},SNR:{SNR}db")
    return wav2, sim, SNR


def f_loss(sim, lables, threshold, confidence=0):
    if lables == 1:
        return torch.clamp(sim - threshold + confidence, min=0)
    else:
        return torch.clamp(threshold - sim + confidence, min=0)


def cw2_attack(
    model,
    wav2,
    emb_t,
    labels,
    loss=torch.nn.CosineEmbeddingLoss(margin=-1),
    confidence=0.5,
    initial_const=1e-3,
    binary_search_steps=9,
    iters=10000,
    stop_early=True,
    stop_early_iter=1000,
    lr=1e-2,
    batch_size=1,
    verbose=1,
    device="cuda",
):
    ori_wav = wav2.data
    wav2 = _to_attack_space(wav2).detach().to(device)
    wav2.requires_grad = True
    const = torch.tensor([initial_const], device=device)
    lower_bound = torch.tensor([0], device=device)
    upper_bound = torch.tensor([1e10], device=device)

    global_best_l2 = torch.tensor([np.infty], device=device)
    global_best_adv_wav = ori_wav.clone()
    global_best_sim = (
        torch.tensor([-2], device=device)
        if labels[0] == -1
        else torch.tensor([2], device=device)
    )

    for _ in range(binary_search_steps):
        modifier = torch.zeros_like(wav2, requires_grad=True)
        optimizer = torch.optim.Adam([modifier], lr=lr)

        best_l2 = torch.tensor([np.infty], device=device)
        best_sim = (
            torch.tensor([-2], device=device)
            if labels[0] == -1
            else torch.tensor([2], device=device)
        )

        continue_flag = True
        prev_loss = np.infty
        for n_iter in range(iters + 1):
            if not continue_flag:
                break
            input_wav = _to_model_space(wav2 + modifier)
            feat2 = Fbank(input_wav).unsqueeze(0).to(device)
            outputs = model(feat2)
            emb_s = outputs[-1] if isinstance(outputs, tuple) else outputs
            model.zero_grad()
            sim = similarity(emb_t, emb_s).to(device)
            print(sim)
            loss1 = f_loss(sim, labels, threshold=0.25, confidence=confidence)
            loss2 = torch.sum(torch.square(input_wav - wav2)).to(device)
            cost = const * loss1 + loss2

            if n_iter < iters:
                cost.backward(retain_graph=True)
                optimizer.step()
                modifier.grad.zero_()

            if stop_early and n_iter % stop_early_iter == 0:
                if cost > 0.9999 * prev_loss:
                    print("Early Stop ! ")
                    continue_flag = False
                prev_loss = cost
            if (labels[0] == 1 and sim <= 0.25) or (labels[0] == -1 and sim >= 0.25):
                if loss2 < best_l2:
                    best_l2 = loss2
                    best_sim = sim
                if (
                    loss2 < global_best_l2
                ):  # l1 <= 0 indicates the attack succeed with at least kappa confidence
                    global_best_l2 = loss2
                    global_best_sim = sim
                    global_best_adv_wav = input_wav

            if (
                best_sim != -2
            ):  # y_pred != -2 infers that IF-BRANCH-1 is entered at least one time, thus the attack succeeds
                upper_bound = min(upper_bound, const)
                if upper_bound < 1e9:
                    const = (lower_bound + upper_bound) / 2
            else:
                lower_bound = max(lower_bound, const)
                if upper_bound < 1e9:
                    const = (lower_bound + upper_bound) / 2
                else:
                    const *= 10
    wav2 = global_best_adv_wav.cpu()
    noise = torch.sub(ori_wav, wav2)
    P = lambda x: torch.sum(torch.pow(x, 2))
    SNR = round(10 * math.log10(P(ori_wav) / P(noise)), 2)
    feat2 = Fbank(wav2).unsqueeze(0).to(device)
    outputs = model(feat2)
    emb_s = outputs[-1] if isinstance(outputs, tuple) else outputs
    sim = similarity(emb_t, emb_s)
    # print(f"loss:{cost.item()},cos sim:{sim.item()},SNR:{SNR}db")
    return wav2, sim, SNR


def fgsm_attack(
    model,
    wav1,
    emb_t,
    labels,
    loss=torch.nn.CosineEmbeddingLoss(margin=-1),
    eps=6400,
    device="cuda",
):
    ori_wav = wav1.data.clone()

    wav1.requires_grad = True

    feat1 = Fbank(wav1).unsqueeze(0).to(device)
    outputs = model(feat1)
    emb_s = outputs[-1] if isinstance(outputs, tuple) else outputs

    cost = loss(emb_s, emb_t, labels).to(device)

    model.zero_grad()
    cost.backward()

    # 获取音频的梯度
    grad = wav1.grad.data

    # 应用FGSM方法：使用梯度符号来生成对抗样本
    adv_wav = wav1 + eps * grad.sign()

    # 将对抗样本的值限制在有效范围内
    adv_wav = torch.clamp(adv_wav, min=-32768, max=32767)

    # 计算SNR
    noise = ori_wav - adv_wav
    P = lambda x: torch.sum(torch.pow(x, 2))
    SNR = round(10 * math.log10(P(ori_wav) / P(noise)), 2)

    # 可选：重新缩放和计算最终特征和相似度（如果需要）
    adv_wav = adv_wav / (1 << 15)
    feat1 = Fbank(adv_wav).unsqueeze(0).to(device)
    outputs = model(feat1)
    emb_s = outputs[-1] if isinstance(outputs, tuple) else outputs
    sim = similarity(emb_t, emb_s)

    # 返回修改后的音频，相似度，以及SNR
    return adv_wav, sim, SNR


def rfgsm_attack(
    model,
    wav1,
    emb_t,
    labels,
    loss=torch.nn.CosineEmbeddingLoss(margin=-1),
    eps=6400,
    alpha=3,
    device="cuda",
):
    ori_wav = wav1.data.clone().to(device)

    # 首先对输入样本添加一个小的随机扰动
    # alpha通常比eps小，这里使用一个较小的扰动
    wav1 = wav1.to(device)
    noise = alpha * torch.randn(wav1.shape).to(device)
    wav1 = wav1 + noise
    wav1 = torch.clamp(wav1, min=-32768, max=32767)

    wav1.requires_grad = True

    feat1 = Fbank(wav1).unsqueeze(0).to(device)
    outputs = model(feat1)
    emb_s = outputs[-1] if isinstance(outputs, tuple) else outputs

    cost = loss(emb_s, emb_t, labels).to(device)

    model.zero_grad()
    cost.backward()

    grad = wav1.grad.data

    # 应用FGSM方法：使用梯度符号来生成对抗样本
    adv_wav = wav1 + eps * grad.sign()

    # 将对抗样本的值限制在有效范围内
    adv_wav = torch.clamp(adv_wav, min=-32768, max=32767)

    # 计算SNR
    noise = ori_wav - adv_wav
    P = lambda x: torch.sum(torch.pow(x, 2))
    SNR = round(10 * math.log10(P(ori_wav) / P(noise)), 2)

    # 可选：重新缩放和计算最终特征和相似度（如果需要）
    adv_wav = adv_wav / (1 << 15)
    feat1 = Fbank(adv_wav).unsqueeze(0).to(device)
    outputs = model(feat1)
    emb_s = outputs[-1] if isinstance(outputs, tuple) else outputs
    sim = similarity(emb_t, emb_s)

    # 返回修改后的音频，相似度，以及SNR
    return adv_wav, sim, SNR


def ifgsm_attack(
    model,
    wav1,
    emb_t,
    labels,
    loss=torch.nn.CosineEmbeddingLoss(margin=-1),
    eps=6400,
    alpha=3,
    iters=50,
    device="cuda",
):
    ori_wav = wav1.data.clone().to(device)

    # 扰动初始化为0
    perturbation = torch.zeros_like(wav1).to(device)

    wav1 = wav1.to(device)
    emb_t = emb_t.to(device)
    labels = labels.to(device)

    for i in range(iters):
        # 设置requires_grad属性为True，以便计算梯度
        wav1.requires_grad = True

        feat1 = Fbank(wav1).unsqueeze(0).to(device)
        outputs = model(feat1)
        emb_s = outputs[-1] if isinstance(outputs, tuple) else outputs

        cost = loss(emb_s, emb_t, labels).to(device)

        # 反向传播计算梯度
        model.zero_grad()
        # 只有在最后一次迭代之前保留计算图
        cost.backward(retain_graph=(i < iters - 1))

        # 获取音频的梯度符号
        grad_sign = wav1.grad.data.sign()

        # 更新扰动
        perturbation += alpha * grad_sign
        perturbation = torch.clamp(perturbation, min=-eps, max=eps)

        wav1 = ori_wav + perturbation
        wav1 = torch.clamp(wav1, min=-32768, max=32767)
        wav1 = wav1.detach()
        wav1.requires_grad = False

    # 计算SNR
    noise = ori_wav - wav1
    P = lambda x: torch.sum(torch.pow(x, 2))
    SNR = round(10 * math.log10(P(ori_wav) / P(noise)), 2)

    # 可选：重新缩放和计算最终特征和相似度（如果需要）
    wav1 = wav1 / (1 << 15)
    feat1 = Fbank(wav1).unsqueeze(0).to(device)
    outputs = model(feat1)
    emb_s = outputs[-1] if isinstance(outputs, tuple) else outputs
    sim = similarity(emb_t, emb_s)

    # 返回修改后的音频，相似度，以及SNR
    return wav1, sim, SNR


def bim_attack(
    model,
    wav2,
    emb_t,
    labels,
    loss=torch.nn.CosineEmbeddingLoss(margin=-1),
    eps=30,
    alpha=1,
    iters=50,
    device="cuda",
):
    ori_wav = wav2.data
    for _ in range(iters):
        wav2.requires_grad = True
        feat2 = Fbank(wav2).unsqueeze(0).to(device)
        outputs = model(feat2)
        emb_s = outputs[-1] if isinstance(outputs, tuple) else outputs
        model.zero_grad()
        # print(emb_s, emb_t, labels)
        cost = loss(emb_s, emb_t, labels).to(device)
        grad = torch.autograd.grad(cost, wav2, retain_graph=False, create_graph=False)[
            0
        ]
        adv_wav = wav2.detach() + alpha * grad.sign()
        a = torch.clamp(ori_wav - eps, min=0)
        b = (adv_wav >= a).float() * adv_wav + (adv_wav < a).float() * a
        c = (b > ori_wav + eps).float() * (ori_wav + eps) + (
            b <= ori_wav + eps
        ).float() * b
        wav2 = torch.clamp(c, min=-32768, max=32767).detach()
    noise = torch.sub(ori_wav, wav2)
    P = lambda x: torch.sum(torch.pow(x, 2))
    SNR = round(10 * math.log10(P(ori_wav) / P(noise)), 2)
    feat2 = Fbank(wav2).unsqueeze(0).to(device)
    outputs = model(feat2)
    emb_s = outputs[-1] if isinstance(outputs, tuple) else outputs
    sim = similarity(emb_t, emb_s)
    # print(f"loss:{cost.item()},cos sim:{sim.item()},SNR:{SNR}db")
    return wav2, sim, SNR

def pgd_2_old(
    model,
    wav1,
    emb_t,
    labels,
    loss=torch.nn.CosineEmbeddingLoss(margin=-1),
    eps=6400,
    eps_for_division=1e-10,
    alpha=1,
    iters=50,
    device='cuda',
):
    ori_wav = wav1.data
    for _ in range(iters):
        wav1.requires_grad = True
        feat1 = Fbank(wav1).unsqueeze(0).to(device)
        outputs = model(feat1)
        emb_s = outputs[-1] if isinstance(outputs, tuple) else outputs
        model.zero_grad()
        cost = loss(emb_s, emb_t, labels).to(device)
        grad = torch.autograd.grad(cost,wav1,retain_graph=False,create_graph=False)[0]
        grad_norms = (
                torch.norm(grad.view(1, -1), p=2, dim=1)
                + eps_for_division
            )  
        grad = grad / grad_norms
        adv_wav = wav1.detach() + alpha * grad.sign()
        eta = adv_wav - ori_wav
        eta_norms = torch.norm(eta.view(1, -1), p=2, dim=1)
        # print(eta_norms)
        factor = eps/eta_norms
        factor = torch.min(factor, torch.ones_like(eta_norms))
        eta = eta * factor.view(-1, 1,)
        # print(eta)
        wav1 = torch.clamp(ori_wav + eta, min=-32768, max=32767).detach()
    noise = torch.sub(ori_wav, wav1)
    P = lambda x: torch.sum(torch.pow(x, 2))
    SNR = round(10 * math.log10(P(ori_wav) / P(noise)), 2)
    feat1 = Fbank(wav1).unsqueeze(0).to(device)
    outputs = model(feat1)
    emb_s = outputs[-1] if isinstance(outputs, tuple) else outputs
    sim = similarity(emb_t, emb_s)
    # print(f"loss:{cost.item()},cos sim:{sim.item()},SNR:{SNR}db")
    return wav1, sim, SNR
