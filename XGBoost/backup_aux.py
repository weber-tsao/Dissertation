# -*- coding: utf-8 -*-
"""
Created on Sun Jul 31 22:28:59 2022

@author: Asus
"""

aux_model = LogisticRegression(68,1)

awl = AutomaticWeightedLoss(2)	# we have 2 losses
loss_1 = nn.MSELoss()
loss_2 = nn.MSELoss()

# learnable parameters
optimizer = optim.Adam([
                {'params': aux_model.parameters()},
                {'params': awl.parameters(), 'weight_decay': 0}
            ])

for i in range(10):
    for d, label1, label2 in zip(data, data_label, array):
        # forward
        d = torch.from_numpy(d)
        pred1, pred2 = aux_model(d.float()),aux_model(d.float())
        #print(pred1.size())
        #print(type(np.array(label1)))
        #print(type(pred1))
        # calculate losses
        loss1 = loss_1(pred1, torch.from_numpy(np.array(label1)).float())
        loss2 = loss_2(pred2, torch.from_numpy(np.array(label2)).float())
        # weigh losses
        loss_sum = awl(loss1, loss2)
        # backward
        optimizer.zero_grad()
        loss_sum.backward()
        optimizer.step()
        #print('epoch {}, loss {}'.format(i, loss_sum))

d = torch.from_numpy(data)
predicted = aux_model(d.float())
predicted_cls = predicted.round()
data_label_test = torch.from_numpy(data_label.astype(np.float32))
print(type(data_label_test))
acc = predicted_cls.eq(data_label_test).sum() / float(data_label_test.shape[0])
print(f'accuracy = {acc: .4f}')