import torch.nn as nn

def buildNetwork(layers, dropouts, activation=nn.ReLU()):
    net = []
    for i in range(1, len(layers)):
        if dropouts[i-1] > 0:
            net.append(nn.Dropout(dropouts[i-1]))
        net.append(nn.Linear(layers[i - 1], layers[i]))
        if i < len(layers) - 1:
            net.append(activation)
    net = nn.Sequential(*net)
    return net

class AnnotationHead(nn.Module):
    def __init__(self, in_dim, hidden_dim, num_classes, num_layers, dropout, norm, batch_num, **kwargs):
        super().__init__()
        self.ce_loss = nn.CrossEntropyLoss()
        layers = [in_dim] + [hidden_dim] * (num_layers - 1) + [num_classes] 
        dropouts = [dropout] * len(layers)
        print(layers)
        self.mlp = buildNetwork(layers, dropouts)

    def forward(self, x_dict):
        logits = self.mlp(x_dict['h'][x_dict['loss_mask']])
        pred = logits.argmax(1)
        y = x_dict['label'][x_dict['loss_mask']].long()
        loss = self.ce_loss(logits, y)
        return {'pred': pred, 'latent': x_dict['h'], 'label': y}, loss
    
class ModalityPredHead(nn.Module):
    def __init__(self, in_dim, hidden_dim, num_classes, num_layers, dropout, norm, batch_num, **kwargs):
        super().__init__()
        self.ce_loss = nn.MSELoss()
        layers = [in_dim] + [hidden_dim] * (num_layers - 1) + [num_classes] 
        dropouts = [dropout] * len(layers)
        self.mlp = buildNetwork(layers, dropouts)

    def forward(self, x_dict):
        if 'loss_mask' in x_dict:
            preds = self.mlp(x_dict['h'][x_dict['loss_mask']])
            # pred = logits.argmax(1)
            y = x_dict['label'][x_dict['loss_mask']]
            loss = self.ce_loss(preds, y)
        return {'pred': preds, 'latent': x_dict['h'], 'label': y}, loss



