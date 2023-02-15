from MLSurv import *
from loss import *
from dataload import *
from torch import nn
from lifelines.utils import concordance_index

cuda = True if torch.cuda.is_available() else False
if cuda:
    device = torch.device(f'cuda:{0}' if torch.cuda.is_available() else 'cpu')

input_dim = 512
encoder1 = Encoder(input_dim, 256)
decoder1 = Decoder(64, 256, input_dim)
encoder2 = Encoder(input_dim, 256)
decoder2 = Decoder(64, 256, input_dim)
model = MLSurv(encoder1, decoder1, encoder2, decoder2)
model = model.to(device)

learning_rate = 5e-5
epochs = 500

optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
criterion1 = nn.MSELoss()   # NEED FIX : to VAE loss
criterion2 = nn.CrossEntropyLoss()  # NEED FIX : to SurvLoss

# loss_axis = []
# cindex_train = []
# cindex_test = []

def train_loop(dataloader, model, loss_fn1, loss_fn2, optimizer, t):
    size = len(dataloader.dataset)
    for batch, (X, time, observed) in enumerate(dataloader):
        # Compute prediction and loss
        pred = model(X)
        pred_score = model(X)   # NEED FIX : for calculating c-index
        loss = loss_fn1(pred) + loss_fn2(pred)  # NEED FIX

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        train_perf = 0
        train_perf = concordance_index(event_times = time.cpu().detach().numpy(),
                                       event_observed = observed.cpu().detach().numpy(),
                                       predicted_scores = -pred_score.cpu().detach().numpy())
        
        if ((t+1) % 25 == 0):
            print(f"Epoch {t+1}\n-------------------------------")
            print(f"Train Accuracy: {train_perf}\n")
            print(f"Train Loss: {loss.item()}\n")
        # print(f"Type of loss: {type(loss.item())}") # output: float
        
        # loss_axis.append(loss.item())
        # cindex_train.append(train_perf)

# iterate over test dataset to check model performance
def test_loop(dataloader, model, t):
    size = len(dataloader.dataset)
    test_perf = 0

    with torch.no_grad():
        for X, time, observed in dataloader:
            pred = model.forward(X)
            pred_score = model.forward(X)   # NEED FIX : for calculating c-index
            # print(f"event_times shape: {time.numpy().shape}")
            # print(f"predicted shape: {pred.numpy().shape}\n")
            test_perf = concordance_index(event_times = time.cpu().detach().numpy(),
                                          event_observed = observed.cpu().detach().numpy(),
                                          predicted_scores = -pred_score.cpu().detach().numpy())

    
    if ((t+1) % 50 == 0):
        print(f"Test Accuracy: {test_perf}\n")
        
    # cindex_test.append(test_perf)

for t in range(epochs):
    model.train()
    train_loop(train_dataloader, model, criterion1, criterion2, optimizer, t)
    model.eval()
    test_loop(test_dataloader, model, t)
