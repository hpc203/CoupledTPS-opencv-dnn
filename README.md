训练源码在 https://github.com/nie-lang/CoupledTPS 
它的论文名称是《Semi-Supervised Coupled Thin-Plate Spline Model for Rotation Correction and Beyond》
发表在了顶级期刊 IEEE Transactions on Pattern Analysis and Machine Intelligence 上了
，足以见得它的牛逼程度了。

onnx文件在百度云盘 链接: https://pan.baidu.com/s/1GQbXJ2LMhxHFn2-Q1JNBcw 提取码: bxsi

其中的上采样upsample和grid_sample，没有把这两个算子导入到onnx文件里的，
自己独立编写了C++程序实现了，输入和输出都是4维张量的Mat。在编写C++程序时，需要注意
对于四维Mat的索引访问像素值，不能使用at函数，能使用ptr函数,，而大于四维的Mat，既不能使用at，也不能使用ptr访问元素，
只能使用指针形式访问，例如float型的Mat， float* pdata = (float*)Mat.data;
并且在编写C++程序时发现一个坑，在4维张量转到RGB三通道彩图时，使用ptr函数的方式给像素点赋值，最后得到的结果图跟Python程序运行的结果图不一致。
但是使用指针形式给像素点赋值，最后得到的结果图跟Python程序运行的结果图就是一致的，
看来使用指针形式访问像素值是最稳妥不出错的方式了。这个坑在C++代码里的convert4dtoimage函数里，我在函数里有写注释。
