from .downstream import AnnotationHead,ModalityPredHead
# from .spatial import
from torch import nn

def setup_head(head_type, in_dim, hidden_dim, out_dim, num_layers, dropout, norm, batch_num) -> nn.Module:
    if head_type == 'annotation':
        mod = AnnotationHead(
            in_dim=in_dim,
            hidden_dim=hidden_dim,
            num_classes=out_dim,
            num_layers=num_layers,
            dropout=dropout,
            norm=norm,
            batch_num=batch_num,
        )
    elif head_type == 'predmodality':
        mod = ModalityPredHead(
            in_dim=in_dim,
            hidden_dim=hidden_dim,
            num_classes=out_dim,
            num_layers=num_layers,
            dropout=dropout,
            norm=norm,
            batch_num=batch_num,
        )
    else:
        raise NotImplementedError(f'Unsupported model type: {head_type}')
    return mod