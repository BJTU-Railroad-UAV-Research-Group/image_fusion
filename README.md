## 0.环境配置

创建虚拟环境；

激活虚拟环境

安装第三方库

```cmd
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple -r requirement.txt
```



## 1.抠图样本存放于文件夹

（1）使用`sample_aug`工程生成的`AugSamples`置于当前工程的根目录。

（2）原始图像及其标注的`json`文件存放在`images`文件夹下。

***注意1***：仅支持`json`标注文件格式，其他格式请转化，脚本见`format_trans`文件夹

***注意2***：如果需要抠图样本必须要在原始图像中合理的范围内出现，例如钢结构涂层表面出现脱落，需要使用标注工具对图像进行前景合理区域标注，标注类别命名为`__mask__`，程序会自动处理抠图样本，保证在合理区域内随机出现且不覆盖既有目标。放心，最后生成的融合图像标注文件会自动去除`__mask__`类别！



## 2.修改配置文件

按需修改`config/config.yml`



## 3.执行增强主程序

```python
python main.py
```



## 4.查看结果

查看输出文件夹`output`下的结果