__all__ = [
    "get_assignments",
    "matched_mask_iou",
    "matched_box_iou",
    "box_matched_mask_iou",
]

import torch
from torchvision.ops import box_iou


def get_assignments(cost_matrix, alpha):
    """
    Solve the linear assingment problem with Adam.
    Use L1 regularization on assignment weights to encourage
    a one to one mapping.

    Parameters
    ----------
    cost_matrix : torch.tensor
        Cost matrix for the the assignment problem
    alpha : float
        Regulatization parameter for optimization.
        We minimize alpha*soft_matching_loss + (1-alpha)*||matching_probs||_1

    Returns
    -------
    assignment_cost : float
        The average cost over all assignments
    matches : torch.tensor[int]
        The assignment of each row of the cost matrix.
    """

    with torch.enable_grad():
        match_mat = torch.randn(
            cost_matrix.shape, requires_grad=True, device=cost_matrix.device
        )
        opt = torch.optim.Adam([match_mat], lr=0.1)

        for i in range(50):
            scaled_matches = match_mat.softmax(1)
            loss = (1 - alpha) * (
                scaled_matches * cost_matrix
            ).sum() + alpha * scaled_matches.sum()
            opt.zero_grad()
            loss.backward()
            opt.step()

    matches = match_mat.argmax(1)

    # matched, inverse, counts = torch.unique(matches,return_inverse=True, return_counts=True)
    # TODO eliminate duplicate assignments
    # for i,(m, c) in enumerate(zip(matched, counts)):
    #     if c > 1:
    #         idx inverse==i
    #         row = torch.arange(match_mat.shape[0])[idx]
    #         torch.argmin(cost_matrix[rows,m])

    assignment_cost = (
        -1 * cost_matrix[torch.arange(cost_matrix.shape[0]), matches].mean().item()
    )
    return assignment_cost, matches


def matched_mask_iou(pred_masks, true_masks, return_matches=False, alpha=0.25):
    """
    Parameters
    ----------
    pred_masks : torch.tensor shape (N, 1, H, W)
        Predicted masks from the output of the mask-rcnn.
        Should have already been thresholded.
    true_masks : torch.tensor shape (M, H, W)
        Ground truth masks.
    alpha : float
        Regularization parameter for linear sum assignment.

    Returns
    -------
    assignment_cost : float
        The average cost over all assignments
    matches : torch.tensor[int]
        The assignment of each row of the cost matrix.
    """
    if pred_masks.shape[0] == 0:
        return 0

    p_bool_masks = pred_masks.to(torch.bool).clone().detach()
    t_bool_masks = true_masks.to(torch.bool)[None, ...]

    intersections = (p_bool_masks & t_bool_masks).sum((-1, -2))
    unions = (p_bool_masks | t_bool_masks).sum((-1, -2))
    neg_iou = -intersections / unions

    assignment_cost, matches = get_assignments(neg_iou, alpha)

    if return_matches:
        return assignment_cost, matches

    else:
        return assignment_cost


def matched_box_iou(pred_boxes, true_boxes, return_matches=False, alpha=0.25):
    """
    Parameters
    ----------

    Returns
    -------
    assignment_cost : float
        The average cost over all assignments
    matches : torch.tensor[int]
        The assignment of each row of the cost matrix.
    """
    if pred_boxes.shape[0] == 0:
        return 0

    neg_iou = -box_iou(pred_boxes, true_boxes).clone().detach()
    assignment_cost, matches = get_assignments(neg_iou, alpha)

    if return_matches:
        return assignment_cost, matches

    else:
        return assignment_cost


def box_matched_mask_iou(matches, pred_masks, true_mask):
    aligned_pred_mask = pred_masks.squeeze()[matches]

    intersections = (aligned_pred_mask & true_mask).sum(-1, -2)
    unions = (aligned_pred_mask | true_mask).sum((-1, -2))

    ious = intersections / unions

    return ious.mean().item()
