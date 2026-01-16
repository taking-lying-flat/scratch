def compute_approx_kl(
    log_probs: torch.Tensor,
    log_probs_base: torch.Tensor,
    kl_estimator: str = "k1",
) -> torch.Tensor:
    """
    Compute the approximate KL divergence between two distributions.
    Schulman blog: http://joschu.net/blog/kl-approx.html

    Args:
        log_probs: Log probabilities of the new distribution.
        log_probs_base: Log probabilities of the base distribution.
    """

    if kl_estimator == "k1":
        log_ratio = log_probs.float() - log_probs_base.float()

    # The k2 estimator is the non negative kl approximation in
    # http://joschu.net/blog/kl-approx.html
    # The k2_loss is approximately equivalent to the one-step KL divergence penalty with the k1 estimator
    # used in https://arxiv.org/pdf/2310.10505.
    if kl_estimator == "k2":
        log_ratio = log_probs.float() - log_probs_base.float()
        log_ratio = log_ratio**2 / 2.0

    # The k3 estimator is the non negative kl approximation in
    # http://joschu.net/blog/kl-approx.html
    if kl_estimator == "k3":
        log_ratio = log_probs.float() - log_probs_base.float()
        log_ratio = -log_ratio
        log_ratio = log_ratio.exp() - 1 - log_ratio

    log_ratio = log_ratio.clamp(min=-10, max=10)
    return log_ratio


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int = 512, nums_head: int = 8, dropout: float = 0.0):
        super().__init__()
        assert d_model % nums_head == 0
        self.d_model = d_model
        self.nums_head = nums_head
        self.d_head = d_model // nums_head
        
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None):
        B, T, C = x.shape
        
        qkv = self.qkv(x)                                      # (B, T, 3*C)
        qkv = qkv.view(B, T, 3, self.nums_head, self.d_head)   # (B, T, 3, H, Dh)
        qkv = qkv.permute(2, 0, 3, 1, 4)                       # (3, B, H, T, Dh)
        q, k, v = qkv[0], qkv[1], qkv[2]                       # (B, H, T, Dh)
        
        attn_scores = q @ k.transpose(-2, -1) / math.sqrt(self.d_head)  # (B, H, T, T)
        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask == 0, float('-inf'))
        
        attn = attn_scores.softmax(dim=-1)                     # (B, H, T, T)
        attn = self.dropout(attn)
        out = attn @ v                                         # (B, H, T, Dh)
        
        out = out.transpose(1, 2).contiguous().view(B, T, C)   # (B, T, C)
        out = self.out_proj(out)
        return out


def count_tokens(text):
    encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")  # Using a default model
    return len(encoding.encode(text))
