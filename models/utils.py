import torch
from torch.nn.functional import cross_entropy


def contrastive_loss(projections, temperature=0.1):
    """
    Compute NT-Xent loss for contrastive learning.
    
    Args:
        projections (torch.Tensor): Tensor of shape [2N, d], where 2N is the batch size
                                    (N samples with two augmented views each) and d is
                                    the feature dimension.
        temperature (float): Temperature scaling parameter for the loss.
    
    Returns:
        torch.Tensor: Scalar loss value.
    """
    # Normalize projections to unit vectors
    # projections = F.normalize(projections, dim=1)

    # Compute similarity matrix (cosine similarity)
    similarity_matrix = torch.matmul(projections, projections.T)  # Shape: [2N, 2N]
    
    # Scale by temperature
    similarity_matrix = similarity_matrix / temperature

    # Create labels for positive pairs
    N = projections.size(0) // 2  # Number of original samples
    labels = torch.add(torch.arange(2*N, device=projections.device), torch.tensor([1, -1], device=projections.device).repeat(N))

    # Mask out self-similarities
    mask = torch.eye(2 * N, device=projections.device).bool()
    similarity_matrix.masked_fill_(mask, float('-inf'))

    # Compute loss using cross-entropy
    loss = cross_entropy(similarity_matrix, labels)
    return loss

def l2_normalize(tensor, dim=1, epsilon=1e-12):
    """
    Performs L2 normalization on the given tensor along the specified dimension.
    
    Args:
        tensor (torch.Tensor): The input tensor to normalize.
        dim (int): The dimension along which to normalize (default: 1).
        epsilon (float): Small constant to avoid division by zero (default: 1e-12).
    
    Returns:
        torch.Tensor: L2-normalized tensor.
    """
    norm = torch.sqrt(torch.sum(tensor**2, dim=dim, keepdim=True) + epsilon)
    return tensor / norm

if __name__ == "__main__":
    random_tensor = torch.rand(4, 3)  # A 4x3 tensor
    print("Original Tensor:")
    print(random_tensor)

    # Perform L2 normalization along dimension 1
    normalized_tensor = l2_normalize(random_tensor, dim=1)
    print("\nL2-Normalized Tensor (along dim=1):")
    print(normalized_tensor)

    # Check if the norm of each row is approximately 1
    row_norms = torch.sqrt(torch.sum(normalized_tensor**2, dim=1))
    print("\nNorm of each row after normalization (should be ~1):")
    print(row_norms)

    # Define small test projections manually
    projections = torch.tensor([
        [1.0, 0.0],  # a_1
        [1.0, 0.0],  # a_2
        [0.0, 1.0],  # b_1
        [0.0, 1.0],  # b_2
    ], dtype=torch.float32)

    # Expected similarity matrix (normalized manually)
    # After normalization, vectors remain the same because they are unit vectors
    # Cosine similarities:
    # [a_1, a_2, b_1, b_2]
    # a_1:  1    1    0    0
    # a_2:  1    1    0    0
    # b_1:  0    0    1    1
    # b_2:  0    0    1    1

    # Calculate the loss
    loss = contrastive_loss(projections, temperature=0.1)
    print(f"Contrastive Loss: {loss.item():.4f}")

    # Manual validation:
    # Positive logits: exp(similarity / temp) -> exp(1 / 0.1)
    # Negative logits: exp(similarity / temp) -> exp(0 / 0.1)
    # Cross-entropy should match the logits for positive labels.