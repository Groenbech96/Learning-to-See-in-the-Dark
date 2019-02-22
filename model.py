
import torch
import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()

        # initialize weights and biases
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                m.bias.data.zero_()
            elif isinstance(m, nn.ConvTranspose2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))

        # Define graph
        self.conv1_a = nn.Conv2d(in_channels=4, out_channels=32,  kernel_size=(3,3),
                                stride=(1,1), padding=(1,1), dilation=(1,1))
        self.conv1_b = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(3,3),
                                stride=(1,1), padding=(1,1), dilation=(1,1))
        self.pool1 = nn.MaxPool2d(kernel_size=(2,2))

        self.conv2_a = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3,3),
                                stride=(1,1), padding=(1,1), dilation=(1,1))
        self.conv2_b = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3,3),
                                stride=(1,1), padding=(1,1), dilation=(1,1))
        self.pool2 = nn.MaxPool2d(kernel_size=(2,2))

        self.conv3_a = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(3,3),
                                stride=(1,1), padding=(1,1), dilation=(1,1))
        self.conv3_b = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3,3),
                                stride=(1,1), padding=(1,1), dilation=(1,1))
        self.pool3 = nn.MaxPool2d(kernel_size=(2,2))

        self.conv4_a = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(3,3),
                                stride=(1,1), padding=(1,1), dilation=(1,1))
        self.conv4_b = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(3,3),
                                stride=(1,1), padding=(1,1), dilation=(1,1))

        self.pool4 = nn.MaxPool2d(kernel_size=(2,2))


        self.conv5_a = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=(3,3),
                                stride=(1,1), padding=(1,1), dilation=(1,1))

        self.conv5_b = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=(3,3),
                                stride=(1,1), padding=(1,1), dilation=(1,1))

        self.up6 = nn.ConvTranspose2d(in_channels=512, out_channels=256,
                                      kernel_size=(2,2),stride=(2,2))
        self.conv6_a = nn.Conv2d(in_channels=512, out_channels=256, kernel_size=(3,3),
                                stride=(1,1), padding=(1,1), dilation=(1,1))
        self.conv6_b = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(3,3),
                                stride=(1,1), padding=(1,1), dilation=(1,1))

        self.up7 = nn.ConvTranspose2d(in_channels=256, out_channels=128,
                                      kernel_size=(2,2),stride=(2,2))
        self.conv7_a = nn.Conv2d(in_channels=256, out_channels=128, kernel_size=(3,3),
                                stride=(1,1), padding=(1,1), dilation=(1,1))
        self.conv7_b = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3,3),
                                stride=(1,1), padding=(1,1), dilation=(1,1))

        self.up8 = nn.ConvTranspose2d(in_channels=128, out_channels=64,
                                      kernel_size=(2,2),stride=(2,2))
        self.conv8_a = nn.Conv2d(in_channels=128, out_channels=64, kernel_size=(3,3),
                                stride=(1,1), padding=(1,1), dilation=(1,1))
        self.conv8_b = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3,3),
                                stride=(1,1), padding=(1,1), dilation=(1,1))

        self.up9 = nn.ConvTranspose2d(in_channels=64, out_channels=32,
                                      kernel_size=(2,2),stride=(2,2))
        self.conv9_a = nn.Conv2d(in_channels=64, out_channels=32, kernel_size=(3,3),
                                stride=(1,1), padding=(1,1), dilation=(1,1))
        self.conv9_b = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(3,3),
                                stride=(1,1), padding=(1,1), dilation=(1,1))

        self.conv10 = nn.Conv2d(in_channels=32, out_channels=12, kernel_size=(1,1),
                                stride=(1,1), dilation=(1,1))

        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x):

        # Define forward pass

        c1 = self.lrelu(self.conv1_a(x))
        c1 = self.lrelu(self.conv1_b(c1))
        p1 = self.pool1(c1)

        c2 = self.lrelu(self.conv2_a(p1))
        c2 = self.lrelu(self.conv2_b(c2))
        p2 = self.pool1(c2)

        c3 = self.lrelu(self.conv3_a(p2))
        c3 = self.lrelu(self.conv3_b(c3))
        p3 = self.pool1(c3)

        c4 = self.lrelu(self.conv4_a(p3))
        c4 = self.lrelu(self.conv4_b(c4))
        p4 = self.pool1(c4)

        c5 = self.lrelu(self.conv5_a(p4))
        c5 = self.lrelu(self.conv5_b(c5))

        up6 = self.up6(c5)
        up6 = torch.cat([up6, c4], 1)
        c6 = self.lrelu(self.conv6_a(up6))
        c6 = self.lrelu(self.conv6_b(c6))

        up7 = self.up7(c6)
        up7 = torch.cat([up7, c3], 1)
        c7 = self.lrelu(self.conv7_a(up7))
        c7 = self.lrelu(self.conv7_b(c7))

        up8 = self.up8(c7)
        up8 = torch.cat([up8, c2], 1)
        c8 = self.lrelu(self.conv8_a(up8))
        c8 = self.lrelu(self.conv8_b(c8))

        up9 = self.up9(c8)
        up9 = torch.cat([up9, c1], 1)
        c9 = self.lrelu(self.conv9_a(up9))
        c9 = self.lrelu(self.conv9_b(c9))

        c10 = self.conv10(c9)
        out = nn.functional.pixel_shuffle(c10, 2)
        return out

