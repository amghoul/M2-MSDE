import os
import torch

def mergingDifferentModels(args,model,pretrainedModel_path,log,strict=True):
    if os.path.isfile(pretrainedModel_path):
        state_dict = torch.load(pretrainedModel_path)
        for k in list(state_dict['state_dict'].keys()):
            if "filter" in k or "conv3d_alone" in k:
                del state_dict['state_dict'][k]

        temp= "temp_to_del.pth"
        torch.save({'state_dict': model.state_dict()}, temp)
        state_dict2 = torch.load(temp)
        for k in list(state_dict2['state_dict'].keys()):
            if "filter" not in k and "conv3d_alone" not in k and "conv2d_alone" not in k and "edge_aware_refinements.1" not in k and "edge_aware_refinements.2" not in k:
                state_dict2['state_dict'][k]=state_dict['state_dict'][k]
            
        model.load_state_dict(state_dict2['state_dict'])#,strict=strict)
        #total_train_loss_save = state_dict['total_train_loss']
        log.info("-- pretrained model "+ args.model+" merged --")
    else:
        log.info("No pretrained model "+ args.model+" exists to merge")
        exit()
    log.info('Number of model ('+ args.model+') parameters: {}'.format(sum([p.data.nelement() for p in model.parameters()])))
    return model

def freezingtheModel(args,model,log):
    for name, param in model.named_parameters():
        param.requires_grad=False

    if args.model != 'cf_fact3d':
        parent_countrt=0
        for parent in model.children():
            first_ch_countrt=0
            for first_ch in parent.children():
                if first_ch_countrt ==1:
                    for name, param in first_ch.named_parameters():
                        param.requires_grad=True
                first_ch_countrt +=1
            parent_countrt +=1
    else: #args.model == 'cf_fact3d'
        parent_countrt=0
        for parent in model.children():
            first_ch_countrt=0
            for first_ch in parent.children():
                if first_ch_countrt ==1:
                    for name, param in first_ch.named_parameters():
                        param.requires_grad=True
                first_ch_countrt +=1
            parent_countrt +=1

    #for name, param in model.named_parameters():
    #    print(name,": ",param.requires_grad)
    return model