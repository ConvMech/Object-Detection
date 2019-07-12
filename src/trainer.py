class Trainer:
    """Trainer for FasterRCNN.
    
    Parameters
    ----------
    model: nn.Module
        A model for object detection.
    
    optimizer: torch.optim.Optimizer
        PyTorch Optimizer.
    
    criterion: nn.Module
        Loss function module.
    """
    def __init__(self, model, optimizer, criterion=None):
        self.device = 'cpu'
        self.model = model.to(self.device)
        self.optimizer = optimizer
        self.criterion = criterion

    def epoch_train(self, train_loader):
        self.model.train()
        epoch_loss = 0

        for i, (inputs, targets) in enumerate(train_loader):
            inputs = inputs.to(self.device)

            self.optimizer.zero_grad()

            outputs = self.model(inputs, targets)
            loss_classifier = outputs['loss_classifier']
            loss_box_reg = outputs['loss_box_reg']
            loss_objectness = outputs['loss_objectness']

            loss = loss_classifier + loss_box_reg
            loss.backward()
            self.optimizer.step()
            epoch_loss += loss.item()

        return epoch_loss / len(train_loader)

    def epoch_eval(self, eval_loader):
        self.model.eval()
        epoch_loss = 0

        for i, (inputs, targets) in enumerate(eval_loader):
            inputs = inputs.to(self.device)

            outputs = self.model(inputs)
            loss_classifier = outputs['loss_classifier']
            loss_box_reg = outputs['loss_box_reg']
            loss_objectness = outputs['loss_objectness']

            loss = loss_classifier + loss_box_reg
            epoch_loss += loss.item()

        return epoch_loss / len(eval_loader)