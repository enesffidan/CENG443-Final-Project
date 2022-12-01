


![alt text](https://kornia.readthedocs.io/en/v0.4.1/_images/sphx_glr_gaussian_blur_001.png)


# CENG443 Heterogeneous Parallel Programing - Final project

- **Subject:** Paralleling Gaussian Image Blur with CUDA 
- **Student:** Enes Furkan Fidan - 250201028
- **Instructor:** Işıl Öz



# Problem Definition
In image blurring, a calculation is made for a pixel, including the values of the surrounding pixels. The main problem is that the execution time and CPU need increase due to the fact that the same process is done for all the pixels in the picture one by one.

It is aimed to greatly improve the performance and execution time metrics by implementing this problem in a data-parallel manner and making a gpu based implementation.

## Gaussian Image Blur
In simple terms, convolution is simply the process of taking a small matrix called the kernel and running it over all the pixels in an image. At every pixel, we’ll perform some math operation involving the values in the convolution matrix and the values of a pixel and its surroundings to determine the value for a pixel in the output image.

![alt text](https://miro.medium.com/max/1400/0*5ZACjFtA_b6WFDUn)

To start off, we’ll need the Gaussian function in two dimensions:

![alt text](https://miro.medium.com/max/376/0*Qyt87iKttnjvkxz8)

The values from this function will create the convolution matrix / kernel that we’ll apply to every pixel in the original image. The kernel is typically quite small — the larger it is the more computation we have to do at every pixel.

![alt text](https://datacarpentry.org/image-processing/fig/blur-demo.gif)



# Implementation Details
> ******NOTE:****** This and following parts will implement later

## Data Parallel Approach

## Explanation of Parallelization Steps


# Experimental Works

## Performance Tests

## Execution Tests




# Future Works

# Conclusion

# Reports
Report format: https://www.ieee.org/conferences/publishing/templates.html

