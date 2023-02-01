for epoch in range(epochs):
    model.train()
    for batch, (X, time_batch, event_batch) in enumerate(dataloader):
        # loss = 0
        decoder_,mu1,sigma1,output,z1 = model.forward(X)
        loss = mse_loss(decoder_,X) + \
            latent_loss(model.z_mean, model.z_sigma) + \
            10*surv_loss.forward(risk = output, times = time_batch, events = event_batch, breaks = model.output_intervals.double().to(device))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    model.eval()
    decoder_,mu1,sigma1,output,z1 = model.forward(X1)
    
    probs_by_interval = output.permute(1, 0).detach().cpu().numpy()
    c_index = [concordance_index(event_times=times,
                                     predicted_scores=interval_probs,
                                     event_observed=events)
                   for interval_probs in probs_by_interval]
    recon_error = mse_loss(decoder_,X1).item()
    latent_error = latent_loss(model.z_mean, model.z_sigma).item()
    surv_error = surv_loss.forward(output,time,event,model.output_intervals.double().to(device)).item()
    print("Recon: ", recon_error,\
        "Latent: ", latent_error,\
            "Surv : ", surv_error,\
            "Total :", recon_error + latent_error + surv_error,
            "Cindex : ", np.mean(c_index))
