**这是CAM可视化的代码，但请注意，这个只是一个提供参考的ToyDemo，实际的实现需要以自己设计的模型为准。**

### 使用方法

#### step1.首先需要安装pytorch_cam的库

`pip install grad-cam`

#### step2.将model.pth、model.py文件放到指定文件夹，并更新config的字段

```
parser.add_argument('--model_name', '-m', type=str, default='xxxx')
parser.add_argument('--weight_name', '-w', type=str, default='xxxx')
parser.add_argument('--file_name', '-f', type=str, default='xxx.pth')
```

#### setp3.最后修改你希望可视化的layer，比如

`target_layers = [model.FA_encoder.encoder_rgb_layer4]`

#### step4.运行cam.py，可视化成功的图像会直接保存到RUN文件夹下

*<u>值得注意的是，如果您希望希望实现其他模型的CAM可视化，您需要调整model.py中fuse模块的组成，需要将输出concat为一个feature map</u>*

EAEFNet：

```
New_fuse = New_RGB * rgb_gate + New_T * t_gate
New_RGB = New_RGB * rgb_gate
New_T = New_T * t_gate
out = torch.cat([New_RGB,New_T,New_fuse],dim=1)
```

FEANet：

```
rgb = rgb.mul(self.atten_depth_channel_4_1(rgb))
rgb = rgb.mul(self.atten_depth_spatial_4_1(rgb))
temp = thermal.mul(self.atten_depth_channel_4_1(thermal))
temp = temp.mul(self.atten_depth_spatial_4_1(temp))
fuse = rgb + temp
out = torch.cat([rgb,temp,fuse],dim=1)
```

如果您在下一层需要重复调用rgb，t，还有fuse的话，需要解开feature map，比如

```
rgb_3_1 = out_fuse_3[:, 0:1024, :]
thermal_3_1 = out_fuse_3[:, 1024:2048, :]
fuse_3_1 = out_fuse_3[:, 2048:1024*3, :]
```
