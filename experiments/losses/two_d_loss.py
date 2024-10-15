import torch
from torch import nn
from transformers import Trainer


class Two_D_Repr_Loss(Trainer):
    def __init__(self, model, **kwargs):
        super().__init__(model, **kwargs)
        self.logits_method = None

    def get_logits(self, model_output):
        if self.logits_method is None:
            try:
                _ = model_output[0]
                self.logits_method = "tuple"
            except TypeError:
                self.logits_method = "attribute"

        if self.logits_method == "tuple":
            return model_output[0]
        elif self.logits_method == "attribute":
            return model_output.logits

    def compute_loss(self, model, inputs):
        device = 'cuda'

        input_ids = inputs.pop("input_ids").to(device)
        labels = input_ids.clone().to(device)
        input_ids = input_ids[:, :-1].contiguous()  # Truncate the input_ids
        lm_logits = self.get_logits(model(input_ids))
        m = nn.Softmax(dim=2)
        lm_logits = m(lm_logits)
        shift_labels = labels[:, 1:].contiguous()
        # print(shift_labels[0])

        # Define y_values and x_values
        y_values = (
            torch.tensor(
                [0.001, 0.001, 0.001, -0.8660254037844386, 0.8660254037844386, -0.5, 0.5, 0.001]
            )
            .reshape(1, 1, 8)
            .to(device)
        )
        x_values = (
            torch.tensor(
                [0.001, 0.001, 0.001, 0.5, 0.5, 0.8660254037844386, 0.8660254037844386, 0.001]
            )
            .reshape(1, 1, 8)
            .to(device)
        )

        # Process y_values loss
        lm_logits_y = lm_logits * y_values
        lm_logits_y = lm_logits_y.cumsum(dim=1)

        shift_labels_one_hot = torch.nn.functional.one_hot(shift_labels, num_classes=8).to(device)

        shift_labels_y = (shift_labels_one_hot + torch.Tensor([0.01]).to(device)) * y_values

        # shift_labels_y = (shift_labels_one_hot + torch.Tensor([0.01])) * y_values
        shift_labels_y = shift_labels_y.cumsum(dim=1).reshape(lm_logits_y.shape)

        var_y = torch.ones(
            lm_logits_y.shape, device=device, requires_grad=True
        )  # heteroscedastic variance
        loss_fn_y = nn.GaussianNLLLoss()
        loss_y = loss_fn_y(lm_logits_y, shift_labels_y, var_y)

        # Process x_values loss
        lm_logits_x = lm_logits * x_values
        lm_logits_x = lm_logits_x.cumsum(dim=1)

        shift_labels_x = (shift_labels_one_hot + torch.Tensor([0.01]).to(device)) * x_values

        # shift_labels_x = (shift_labels_one_hot + torch.Tensor([0.01])) * x_values
        shift_labels_x = shift_labels_x.cumsum(dim=1).reshape(lm_logits_x.shape)
        var_x = torch.ones(
            lm_logits_x.shape, device=device, requires_grad=True
        )  # heteroscedastic variance
        loss_fn_x = nn.GaussianNLLLoss()
        loss_x = loss_fn_x(lm_logits_x, shift_labels_x, var_x)

        # loss_fct = torch.nn.CrossEntropyLoss()
        # lm_loss = loss_fct(lm_logits.view(-1, lm_logits.size(-1)), shift_labels.view(-1))
        # # Total loss
        total_loss = loss_y + loss_x  # + lm_loss

        # print("Loss_y:", loss_y)
        # print("Loss_x:", loss_x)
        # print("Total_loss:", total_loss)

        return total_loss
