/******************************************************************************
Program: 	Part D - C implementation for Part B of Homework 1
Course:		BME 595
Author:		Ankit Manerikar
Date: 		08/31/2017
Inst.:		Purdue University
*******************************************************************************/


#include <stdio.h>
#include <opencv2/opencv.hpp>
#include <ctime>
#include <iostream>
#include <fstream>

using namespace cv;
using namespace std;

void get_convolution(Mat image_1, Mat Kernel, int im_no)
/*******************************************************************************
Description: Performs Convolution on the input image for increasing number of 
			 channels (2^i, i= 0,....10)
Args:        image_1 - input image with 3 channels
			 Kernel  - Kernel for Convolution
			 im_no   - Image Number
Reutrns:     - 
*******************************************************************************/
{
    Mat image_1_ch[3];
 	split(image_1, image_1_ch);
	double time_taken[11];
	
	for(int pow_num=0; pow_num < 11; pow_num++)
	{   
	    int no_of_channels = pow(2.0,pow_num);

  		clock_t begin = clock();	    
		
		for(int ch_no=0; ch_no< no_of_channels; ch_no++)
		{
	
			Mat O_Image(image_1.rows, image_1.cols, CV_64FC1);
			
			for(int k=0; k < (sizeof(image_1_ch)/sizeof(image_1_ch[0])); k++ )
			{
				for(int i=1; i< image_1_ch[k].rows-1; i++ )
				{
					for(int j=1; j< image_1_ch[k].cols-1; j++ )
			   		{
			   			Mat OpMat;
			   			
			   			image_1_ch[k].rowRange(i-1,i+2).colRange(j-1,j+2).copyTo(OpMat);
						double sum_mat =0;
						
						for(int k3=0; k3<3;k3++)
						{
							for(int l3=0; l3<3;l3++)
							{
								sum_mat = sum_mat + OpMat.at<double>(k3,l3)*Kernel.at<double>(k3,l3);
							}
							O_Image.at<double>(i,j) = sum_mat;
						} 
			   		}
			 	}
			 }
		}
		
		clock_t end = clock();
  		double elapsed_secs = double(end - begin) / CLOCKS_PER_SEC;
  		time_taken[pow_num] = elapsed_secs;
  		cout << elapsed_secs<<endl;
	} 
	
	char fname[1024];
	sprintf(fname, "./results/part_d_image_%i_time_taken.txt",im_no);
	
  ofstream myfile1;
  myfile1.open (fname);
  for(int i= 0; i< 10;i++)
  {
  myfile1 << time_taken[i]<<endl;
  }
  myfile1.close();
}
/******************************************************************************/


int main(int argc, char** argv )
{

//Loading Input images
cout << "Loading Images ... " << endl;
    if ( argc != 3 )
    {
        printf("usage: Enter only 2 images please!\n");
        return -1;
    }

    Mat image_1,image_2;
    image_1 = imread( argv[1], 1 );
    image_2 = imread( argv[2], 1 );

//Check for validity of images
    if ( !image_1.data || !image_2.data )
    {
        printf("No image data \n");
        return -1;
    }

// Randomly initialize a 3x3 kernel
    Mat Kernel = Mat(3, 3, CV_64FC1);
    randu(Kernel, 0.0, 1.0);

// Performing Part D operation on image 1
cout << "\nImage No:  1" <<endl;
cout << "Performing Convolution for 2^i channels ..." << endl;
cout << "Time Taken: "<< endl;
  get_convolution(image_1,Kernel,1);

// Performing Part D operation on image 2
cout << "Image No:  2" <<endl;
cout << "Performing Convolution for 2^i channels ..." << endl;
cout << "Time Taken: "<< endl;
  get_convolution(image_2,Kernel,2);

cout << "Part D complete"<<endl; 
    return 0;
}
