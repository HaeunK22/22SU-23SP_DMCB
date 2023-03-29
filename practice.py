def train_model(model, patience, n_epochs, optimizer):
    
    # Track training loss
    train_losses = []
    # Track validation loss
    valid_losses = []
    # Track average training loss
    avg_train_losses = []
    # Track average validation loss
    avg_valid_losses = []
    # Track average c-index for training data
    avg_train_acc = []
    # Track average c-index for validation data
    avg_valid_acc = []
    
    ### Delete if not needed ###
    t_vae_losses = []
    t_disentangle_losses = []
    t_surv_losses = []
    v_vae_losses = []
    v_disentangle_losses = []
    v_surv_losses = []
    avg_t_vae_losses = []
    avg_t_disentangle_losses = []
    avg_t_surv_losses = []
    avg_v_vae_losses = []
    avg_v_disentangle_losses = []
    avg_v_surv_losses = []
    
    
    # early_stopping object
    early_stopping = EarlyStopping(patience = patience, verbose = True, path = path + '/checkpointinher.pt')
    
    for epoch in range(1, n_epochs + 1):
        model.train()
        for batch, (X1, X2, time_batch, event_batch) in enumerate(train_dataloader, 1):
            # loss = 0
            # Compute predition and loss
            decoder1_, decoder2_, decoder3_, decoder4_, comm1, spe1, comm2, spe2, mu1, sigma1, mu2, sigma2, output, inputmlp = model.forward(X1, X2)
            # VAE Loss
            vae_loss = mse_loss(decoder1_, X1) + mse_loss(decoder2_, X2) + mse_loss(decoder3_, X1) + mse_loss(decoder4_, X2) + \
                latent_loss(mu1, sigma1) + latent_loss(mu2, sigma2)
            # Disentangle Loss
            disentangle_loss = mse_loss(comm1, comm2)
            # Survival Prediction Loss
            surv_error = surv_loss.forward(risk = output, times = time_batch, events = event_batch, breaks = model.output_intervals.double().to(device))
            # Total loss
            loss = 2*vae_loss + \
                disentangle_loss + \
                200*surv_error
                
            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
            # Check train performance
            train_losses.append(loss.item())
            ### Delete if not needed ###
            t_vae_losses.append(vae_loss.item())
            t_disentangle_losses.append(disentangle_loss.item())
            t_surv_losses.append(surv_error.item())
            
            
        surv = pd.DataFrame(output.detach().cpu().numpy()).T
        durations = torch.flatten(train_time).detach().cpu().numpy()
        events = torch.flatten(train_event).detach().cpu().numpy()
        ev = EvalSurv(surv, durations, events, censor_surv='km')
        train_ctd = ev.concordance_td('antolini')
    
        # validate   
        model.eval()
        for X1, X2, time_batch, event_batch in val_dataloader:
            decoder1_, decoder2_, decoder3_, decoder4_, comm1, spe1, comm2, spe2, mu1, sigma1, mu2, sigma2, output, inputmlp = model.forward(X1, X2)
            # VAE Loss
            vae_loss = mse_loss(decoder1_, X1) + mse_loss(decoder2_, X2) + mse_loss(decoder3_, X1) + mse_loss(decoder4_, X2) + \
                latent_loss(mu1, sigma1) + latent_loss(mu2, sigma2)
            # Disentangle Loss
            disentangle_loss = mse_loss(comm1, comm2)
            # Survival Prediction Loss
            surv_error = surv_loss.forward(risk = output, times = time_batch, events = event_batch, breaks = model.output_intervals.double().to(device))
            # Total Loss
            loss = 2*vae_loss + \
                1*disentangle_loss + \
                200*surv_error
            valid_losses.append(loss.item())
            ### Delete if not needed ###
            v_vae_losses.append(vae_loss.item())
            v_disentangle_losses.append(disentangle_loss.item())
            v_surv_losses.append(surv_error.item())

        surv = pd.DataFrame(output.detach().cpu().numpy()).T
        durations = torch.flatten(val_time).detach().cpu().numpy()
        events = torch.flatten(val_event).detach().cpu().numpy()
        ev = EvalSurv(surv, durations, events, censor_surv='km')
        valid_ctd = ev.concordance_td('antolini')

        train_loss = np.average(train_losses)
        valid_loss = np.average(valid_losses)
        avg_train_losses.append(train_loss)
        avg_valid_losses.append(valid_loss)
        avg_train_acc.append(train_ctd)
        avg_valid_acc.append(valid_ctd)
        
        ### Delete if not needed ###
        t_vae_loss = np.average(t_vae_losses)
        t_disentangle_loss = np.average(t_disentangle_losses)
        t_surv_loss = np.average(t_surv_losses)
        v_vae_loss = np.average(v_vae_losses)
        v_disentangle_loss = np.average(v_disentangle_losses)
        v_surv_loss = np.average(v_surv_losses)
        avg_t_vae_losses.append(t_vae_loss)
        avg_t_disentangle_losses.append(t_disentangle_losses)
        avg_t_surv_losses.append(t_surv_losses)
        avg_v_vae_losses.append(v_vae_losses)
        avg_v_disentangle_losses.append(v_disentangle_losses)
        avg_v_surv_losses.append(v_surv_losses)

        
        epoch_len = len(str(n_epochs))
        # print_msg = (f'[{epoch:>{epoch_len}}/{n_epochs:>{epoch_len}}] ' +
        #              f'train_loss: {train_loss:.5f} ' +
        #              f'valid_loss: {valid_loss:.5f}')
        # print(print_msg)
        #if (epoch % 100 == 0):
        print_msg = (f'[{epoch:>{epoch_len}}/{n_epochs:>{epoch_len}}] \n' +
                f'TRAIN\n' + 
                f'vae : {t_vae_loss:.5f} | disentangle : {t_disentangle_loss:.5f} | surv : {t_surv_loss:.5f} | total loss : {train_loss:.5f}\n' +
                f'VALIDATION\n' +
                f'vae : {v_vae_loss:.5f} | disentangle : {v_disentangle_loss:.5f} | surv : {v_surv_loss:.5f} | total loss : {valid_loss:.5f}')
        print(print_msg)

        # clear lists to track next epoch
        train_losses = []
        valid_losses = []
        ### Delete if not needed ###
        t_vae_losses = []
        t_disentangle_losses = []
        t_surv_losses = []
        v_vae_losses = []
        v_disentangle_losses = []
        v_surv_losses = []

        # make check point when validation loss decreases
        early_stopping(valid_loss, model)

        if early_stopping.early_stop:
            print("Early stopping")
            break

   # best model이 저장되어있는 last checkpoint를 로드한다.
    model.load_state_dict(torch.load(path + '/checkpointinher.pt'))
    # print(f'avg_t_vae_losses : {avg_t_vae_losses}')
    # print(f'avg_t_disentangle_losses : {avg_t_disentangle_losses}')
    # print(f'avg_t_surv_losses : {avg_t_surv_losses}')
    return  model, avg_train_losses, avg_valid_losses, avg_train_acc, avg_valid_acc, avg_t_vae_losses, avg_t_disentangle_losses, avg_t_surv_losses
    
# iterate over test dataset to check model performance
def test_loop(dataloader, model):
    print("---------------- Start test -------------------")
    model.eval()
    with torch.no_grad():
        for X1, X2, time_batch, event_batch in dataloader:
            decoder1_, decoder2_, decoder3_, decoder4_, comm1, spe1, comm2, spe2, mu1, sigma1, mu2, sigma2, output, inputmlp = model.forward(X1, X2)
    
        surv = pd.DataFrame(output.detach().cpu().numpy()).T
        durations = torch.flatten(test_time).detach().cpu().numpy()
        events = torch.flatten(test_event).detach().cpu().numpy()
        ev = EvalSurv(surv, durations, events, censor_surv='km')
        test_ctd = ev.concordance_td('antolini')
        # VAE Loss
        vae_loss = mse_loss(decoder1_, X1) + mse_loss(decoder2_, X2) + mse_loss(decoder3_, X1) + mse_loss(decoder4_, X2) + \
                latent_loss(mu1, sigma1) + latent_loss(mu2, sigma2)
        # Disentangle Loss
        disentangle_loss = mse_loss(comm1, comm2)

        # Survival Prediction Loss
        surv_error = surv_loss.forward(risk = output, times = time_batch, events = event_batch, breaks = model.output_intervals.double().to(device))
        # Total Loss
        loss = 2*vae_loss + \
            disentangle_loss + \
            200*surv_error
        print(f"Test Accuracy : ")
        print(f"VAE loss: {vae_loss:>7f},\
            Disentangle loss: {disentangle_loss:>7f},\
            Surv loss: {surv_error:>7f},\
            Total loss: {loss:>7f},\
            Ctd : {test_ctd}")
    return test_ctd
