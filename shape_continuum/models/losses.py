import torch

__all__ = ["coxph_loss"]


def safe_normalize(x):
    """Normalize risk scores to avoid exp underflowing.
    Note that only risk scores relative to each other matter.
    If minimum risk score is negative, we shift scores so minimum
    is at zero.
    """
    x_min, _ = torch.min(x, dim=0)
    c = torch.zeros(x_min.shape)
    if torch.cuda.is_available():
        c = c.cuda()
    norm = torch.where(x_min < 0, -x_min, c)
    return x + norm


def coxph_loss(event, riskset, predictions):
    """Negative partial log-likelihood of Cox's proportional
    hazards model.
    Parameters
    ----------
    event : torch.Tensor
        Binary vector where 1 indicates an event 0 censoring.
    riskset : torch.Tensor
        Boolean matrix where the `i`-th row denotes the
        risk set of the `i`-th instance, i.e. the indices `j`
        for which the observer time `y_j >= y_i`.
    predictions : torch.Tensor
        The predicted outputs. Must be a rank 2 tensor.
    Returns
    -------
    loss : torch.Tensor
        Scalar loss.
    References
    ----------
    .. [1] Faraggi, D., & Simon, R. (1995).
    A neural network model for survival data. Statistics in Medicine,
    14(1), 73–82. https://doi.org/10.1002/sim.4780140108
    """
    if predictions is None:
        raise ValueError("predictions must not be None.")
    if predictions.dim() != 2:
        raise ValueError("predictions must be a 2D tensor.")
    if predictions.shape[1] != 1:
        raise ValueError("last dimension of predictions ({}) must be 1.".format(predictions.shape[1]))
    if event is None:
        raise ValueError("event must not be None.")
    if predictions.dim() != event.dim():
        raise ValueError(
            "Rank of predictions ({}) must equal rank of event ({})".format(predictions.dim(), event.dim())
        )
    if event.shape[1] != 1:
        raise ValueError("last dimension event ({}) must be 1.".format(event.shape[1]))
    if riskset is None:
        raise ValueError("riskset must not be None.")

    event = event.type(predictions.type())
    riskset = riskset.type(predictions.type())
    predictions = safe_normalize(predictions)

    pred_t = torch.transpose(predictions, 0, 1)

    pred_masked = pred_t * riskset
    amax, _ = torch.max(pred_masked, dim=1, keepdim=True)
    pred_shifted = pred_masked - amax

    exp_masked = torch.exp(pred_shifted) * riskset
    exp_sum = torch.sum(exp_masked, dim=1, keepdim=True)

    rr = amax + torch.log(exp_sum)
    losses = event * (rr - predictions)

    loss = torch.mean(losses)

    return loss
