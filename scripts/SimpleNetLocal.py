import torch
import torch.nn as nn
import numpy as np

class BasicBlock(nn.Module):
    '''
    from timm
    '''
    def __init__(self, inplanes, planes, ker_size, stride=1, downsample=None, cardinality=1, base_width=64,
                 reduce_first=1, dilation=1, first_dilation=None, act_layer=nn.LeakyReLU,   norm_layer=nn.BatchNorm1d, attn_layer=None, aa_layer=None, drop_block=None, drop_path=None):
        super(BasicBlock, self).__init__()

        self.conv1 = nn.Conv1d(
            inplanes, planes, kernel_size=ker_size, stride=stride, padding=first_dilation,
            dilation=dilation, bias=True)
        self.bn1 = norm_layer(planes)
        self.act1 = act_layer(inplace=True)
        #self.aa = aa_layer(channels=first_planes, stride=stride) if use_aa else None

        self.conv2 = nn.Conv1d(
            planes, planes, kernel_size=ker_size, padding=ker_size//2, dilation=dilation, bias=True)
        self.bn2 = norm_layer(planes)

        #self.se = create_attn(attn_layer, outplanes)

        self.act2 = act_layer(inplace=True)
        if downsample is not None:
            self.downsample = nn.Sequential(
                nn.Conv1d(inplanes, planes,  stride=stride, kernel_size=ker_size, padding=first_dilation, dilation=dilation, bias=True),
                norm_layer(planes), 
            )
        else: self.downsample=None
        self.stride = stride
        self.dilation = dilation
        self.drop_block = drop_block
        self.drop_path = drop_path

    def zero_init_last_bn(self):
        nn.init.zeros_(self.bn2.weight)

    def forward(self, x):
        #print("x after 0", x.shape)
        shortcut = x

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.act1(x)
        #print("x after 1", x.shape)
        x = self.conv2(x)
        x = self.bn2(x)
        #print("x after 2", x.shape)
        if self.downsample is not None:
            shortcut = self.downsample(shortcut)
        x += shortcut
        x = self.act2(x)
        #print("x after 3", x.shape)
        return x

class WavEncoder(nn.Module):
    def __init__(self, out_dim):
        super().__init__() #128*1*140844 
        self.out_dim = out_dim
        self.feat_extractor = nn.Sequential( #b = (a+3200)/5 a 
                BasicBlock(1, 32, 15, 5, first_dilation=1600, downsample=True),
                BasicBlock(32, 32, 15, 6, first_dilation=0, downsample=True),
                BasicBlock(32, 32, 15, 1, first_dilation=7, ),
                BasicBlock(32, 64, 15, 6, first_dilation=0, downsample=True),
                BasicBlock(64, 64, 15, 1, first_dilation=7),
                BasicBlock(64, 128, 15, 6,  first_dilation=0,downsample=True),     
            )
        
    def forward(self, wav_data):
        wav_data = wav_data.unsqueeze(1)  # add channel dim
        out = self.feat_extractor(wav_data)
        return out.transpose(1, 2)  # to (batch x seq x dim)

class FaceGenerator(nn.Module):
    def __init__(self, facial_dims = 51, audio_f = 128, hidden_size = 256, n_layer = 4, dropout_prob = 0.3):
        super().__init__()
        #self.pre_length = args.pre_frames #4
        #self.gen_length = args.facial_length - args.pre_frames #30
        self.facial_dims = facial_dims
        #self.speaker_f = args.speaker_f
        self.audio_f = audio_f
        #self.facial_in = int(args.facial_rep[-2:])
        
        self.in_size = self.audio_f + self.facial_dims + 1
        self.audio_encoder = WavEncoder(self.audio_f)
        
        self.hidden_size = hidden_size
        self.n_layer = n_layer
        self.dropout_prob = dropout_prob

        # if self.facial_f is not 0:  
        #     self.facial_encoder = nn.Sequential( #b = (a+3200)/5 a 
        #         BasicBlock(self.facial_in, self.facial_f//2, 7, 1, first_dilation=3,  downsample=True),
        #         BasicBlock(self.facial_f//2, self.facial_f//2, 3, 1, first_dilation=1,  downsample=True),
        #         BasicBlock(self.facial_f//2, self.facial_f//2, 3, 1, first_dilation=1, ),
        #         BasicBlock(self.facial_f//2, self.facial_f, 3, 1, first_dilation=1,  downsample=True),   
        #     )
        # else:
        #     self.facial_encoder = None

        
        self.gru = nn.GRU(self.in_size, hidden_size=self.hidden_size, num_layers=self.n_layer, batch_first=True,
                          bidirectional=True, dropout=self.dropout_prob)
        self.out = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size//2),
            nn.LeakyReLU(True),
            nn.Linear(self.hidden_size//2, self.facial_dims)
        )
        
        self.do_flatten_parameters = False
        if torch.cuda.device_count() > 1:
            self.do_flatten_parameters = True
            

    def forward(self, pre_seq, in_audio, is_test=False): #pre_seq in this case is in_face
        decoder_hidden = decoder_hidden_hands = None
        
        if self.do_flatten_parameters:
            self.gru.flatten_parameters()

        audio_feat_seq = None
        audio_feat_seq = self.audio_encoder(in_audio)  # output (bs, n_frames, feat_size)
        
        in_data = torch.cat((pre_seq, audio_feat_seq), dim=-1)
        
        # if speaker_feat_seq is not None:
        #     #if print(z_context.shape)
        #     repeated_s = speaker_feat_seq
        #     #print(repeated_s.shape)
        #     if len(repeated_s.shape) == 2:
        #         repeated_s = repeated_s.reshape(1, repeated_s.shape[1], repeated_s.shape[0])
        #         #print(repeated_s.shape)
        #     repeated_s = repeated_s.repeat(1, in_data.shape[1], 1)
        #     #print(repeated_s.shape)
        #     #print(repeated_s.shape)
        #     in_data = torch.cat((in_data, repeated_s), dim=2)
        
        
        output, decoder_hidden = self.gru(in_data, decoder_hidden)
        output = output[:, :, :self.hidden_size] + output[:, :, self.hidden_size:]  # sum bidirectional outputs
        output = self.out(output.reshape(-1, output.shape[2]))
        decoder_outputs = output.reshape(in_data.shape[0], in_data.shape[1], -1)


        return decoder_outputs

class RMSLELoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()
        
    def forward(self, pred, actual):
        return torch.sqrt(self.mse(torch.log(pred + 1), torch.log(actual + 1)))
    

if __name__ == "__main__":
    from Dataset import a2bsDataset
    from Dataset import a2bsDataset
    train_data = a2bsDataset(build_cache=False, mount_dir='..', data_dir='datasets')

    mount_dir = '..'

    train_loader = torch.utils.data.DataLoader(
            train_data, 
            batch_size=64,  
            shuffle=True,  
            num_workers=0,
            drop_last=True,
        )

    net = FaceGenerator().cuda()
    optimizer = torch.optim.Adam( net.parameters(), lr=1e-3)
    loss_function = RMSLELoss()
    train_loss = []
    eval_loss = []

    net.train()
    num_epochs = 20  #20
    log_period = 100
    eval_period = 400
    eval_it = 30

    for epoch in range(num_epochs):
        for it, (in_audio, facial, in_id) in enumerate(train_loader):
            print(in_audio.shape, facial.shape, in_id.shape)
            net.train()
            in_audio = in_audio.cuda()
            facial = facial.cuda()
            pre_frames = 4
            in_pre_face = facial.new_zeros((facial.shape[0], facial.shape[1], facial.shape[2] + 1)).cuda()
            in_pre_face[:, 0:pre_frames, :-1] = facial[:, 0:pre_frames]
            in_pre_face[:, 0:pre_frames, -1] = 1 
            
            optimizer.zero_grad()
            out_face = net(in_pre_face,in_audio)
            loss = loss_function(facial, out_face)
            loss.backward()
            train_loss.append(loss.item())
            optimizer.step()
            
            #logging
            if it % log_period == 0:
                print(f'[{epoch}][{it}/{len(train_loader)}] loss: {loss.item()}')
            
            if it % eval_period == 0:
                eval_data = a2bsDataset(loader_type='eval', build_cache=False, mount_dir='..', data_dir='datasets')
                eval_loader = torch.utils.data.DataLoader(
                            eval_data, 
                            batch_size=64,  
                            shuffle=True,  
                            num_workers=0,
                            drop_last=True,
                        )
                net.eval()
                eval_loss_st = []
                for i, (in_audio, facial, in_id) in enumerate(eval_loader):
                    in_audio = in_audio.cuda()
                    facial = facial.cuda()
                    pre_frames = 4
                    in_pre_face = facial.new_zeros((facial.shape[0], facial.shape[1], facial.shape[2] + 1)).cuda()
                    in_pre_face[:, 0:pre_frames, :-1] = facial[:, 0:pre_frames]
                    in_pre_face[:, 0:pre_frames, -1] = 1 

                    out_face = net(in_pre_face,in_audio)
                    loss = loss_function(facial, out_face)
                    eval_loss_st.append(loss.item())

                    if i >= eval_it:
                        break
                
                eval_loss.append(np.average(eval_loss_st))
                print(f'[{epoch}][{it}/{len(train_loader)}] eval loss: {np.average(eval_loss_st)}')
        torch.save(net.state_dict(), f'{mount_dir}/ckpt_model/simplenet_ep_{epoch}.pth')
