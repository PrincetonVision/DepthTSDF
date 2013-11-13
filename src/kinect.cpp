/*
Copyright (c) 2011-2013 Gerhard Reitmayr, TU Graz

Permission is hereby granted, free of charge, to any person obtaining a
copy of this software and associated documentation files (the
"Software"), to deal in the Software without restriction, including
without limitation the rights to use, copy, modify, merge, publish,
distribute, sublicense, and/or sell copies of the Software, and to
permit persons to whom the Software is furnished to do so, subject to
the following conditions:

The above copyright notice and this permission notice shall be included
in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
*/

#include "kfusion.h"
#include "helpers.h"
#include "interface.h"
#include "perfstats.h"

#include <iostream>
#include <sstream>
#include <iomanip>
#include <cstring>
#include <dirent.h>
#include <cerrno>

#include <png++/png.hpp>
#include <jpeglib.h>

#ifdef __APPLE__
#include <GLUT/glut.h>
#elif defined(WIN32)
#define GLUT_NO_LIB_PRAGMA
#include <glut.h>
#else
#include <GL/glut.h>
#endif

using namespace std;
using namespace TooN;

KFusion kfusion;
Image<uchar4, HostDevice> lightScene, trackModel, lightModel, texModel;
Image<uint16_t, HostDevice> depthImage[2];
Image<uchar3, HostDevice> rgbImage;

const float3 light = make_float3(1, 1, -1.0);
const float3 ambient = make_float3(0.1, 0.1, 0.1);

SE3<float> initPose;

float size;
int counter = 0;
int integration_rate = 2;
bool reset = true;
bool should_integrate = true;
bool render_texture = false;

Image<float3, Device> pos, normals;
Image<float, Device> dep;

SE3<float> preTrans, trans, rot(makeVector(0.0, 0, 0, 0, 0, 0));
bool redraw_big_view = false;

/*============================================================================*/
#ifdef SUN3D

int file_index = 0;
//int file_index = 1250;

const int kImageRows = 480;
const int kImageCols = 640;
const int kImageChannels = 3;

vector<string> image_list;
vector<string> depth_list;

string data_dir = "/home/alan/DATA/SUN3D/hotel_umd/maryland_hotel3/";
string intrinsic = data_dir + "intrinsics.txt";
string image_dir = data_dir + "image/";
string depth_dir = data_dir + "depth/";

void GetFileNames(const string dir, vector<string> *file_list) {
  DIR *dp;
  struct dirent *dirp;
  if((dp  = opendir(dir.c_str())) == NULL) {
      cout << "Error(" << errno << ") opening " << dir << endl;
  }

  while ((dirp = readdir(dp)) != NULL) {
      file_list->push_back(dir + string(dirp->d_name));
  }
  closedir(dp);

  sort( file_list->begin(), file_list->end() );
  file_list->erase(file_list->begin()); //.
  file_list->erase(file_list->begin()); //..
}

bool GetDepthData(string file_name, uint16_t *data) {
  png::image< png::gray_pixel_16 > img(file_name.c_str(),
      png::require_color_space< png::gray_pixel_16 >());

  int index = 0;
  for (int i = 0; i < kImageRows; ++i) {
    for (int j = 0; j < kImageCols; ++j) {
      uint16_t c = img.get_pixel(j, i);
      *(data + index) = (c >> 3);
//      *(data + index) = (c << 13 | c >> 3);
      ++index;
    }
  }

  return true;
}

bool GetImageData(string file_name, unsigned char *data) {
  unsigned char *raw_image = NULL;

  struct jpeg_decompress_struct cinfo;
  struct jpeg_error_mgr jerr;
  JSAMPROW row_pointer[1];

  FILE *infile = fopen(file_name.c_str(), "rb");
  unsigned long location = 0;

  if (!infile) {
    printf("Error opening jpeg file %s\n!", file_name.c_str());
    return -1;
  }
  cinfo.err = jpeg_std_error(&jerr);
  jpeg_create_decompress(&cinfo);
  jpeg_stdio_src(&cinfo, infile);
  jpeg_read_header(&cinfo, TRUE);
  jpeg_start_decompress(&cinfo);

  raw_image = (unsigned char*) malloc(
      cinfo.output_width * cinfo.output_height * cinfo.num_components);
  row_pointer[0] = (unsigned char *) malloc(
      cinfo.output_width * cinfo.num_components);
  while (cinfo.output_scanline < cinfo.image_height) {
    jpeg_read_scanlines(&cinfo, row_pointer, 1);
    for (uint i = 0; i < cinfo.image_width * cinfo.num_components; i++)
      raw_image[location++] = row_pointer[0][i];
  }

  int index = 0;
  for (uint i = 0; i < cinfo.image_height; ++i) {
    for (uint j = 0; j < cinfo.image_width; ++j) {
      for (int k = 0; k < kImageChannels; ++k) {
        *(data + index) = raw_image[(i * cinfo.image_width * 3) + (j * 3) + k];
        ++index;
      }
    }
  }

  jpeg_finish_decompress(&cinfo);
  jpeg_destroy_decompress(&cinfo);
  free(row_pointer[0]);
  fclose(infile);

  return true;
}

#endif
/*============================================================================*/

void display(void){
    static bool integrate = true;

/*============================================================================*/
#ifdef SUN3D

    cout << file_index << endl;

    if (file_index == 1870)
      exit(0);

    GetImageData(image_list[file_index], (unsigned char *)rgbImage.data());
    GetDepthData(depth_list[file_index], (uint16_t *)depthImage[0].data());
    ++file_index;

    glClear( GL_COLOR_BUFFER_BIT );
    const double startFrame = Stats.start();
    const double startProcessing = Stats.sample("kinect");

    kfusion.setKinectDeviceDepth(depthImage[0].getDeviceImage());
    Stats.sample("raw to cooked");

/*============================================================================*/
#else

    glClear( GL_COLOR_BUFFER_BIT );
    const double startFrame = Stats.start();
    const double startProcessing = Stats.sample("kinect");

    kfusion.setKinectDeviceDepth(depthImage[GetKinectFrame()].getDeviceImage());
    Stats.sample("raw to cooked");
#endif

    integrate = kfusion.Track();
    Stats.sample("track");

    if(!integrate && file_index != 1)
      exit(0);

    if((should_integrate && integrate && ((counter % integration_rate) == 0)) || reset){
        kfusion.Integrate();
        kfusion.Raycast();
        Stats.sample("integrate");
        if(counter > 2) // use the first two frames to initialize
            reset = false;
    }

    renderLight( lightScene.getDeviceImage(), kfusion.inputVertex[0], kfusion.inputNormal[0], light, ambient );
    renderLight( lightModel.getDeviceImage(), kfusion.vertex, kfusion.normal, light, ambient);
    renderTrackResult(trackModel.getDeviceImage(), kfusion.reduction);
    static int count = 4;
    if(count > 3 || redraw_big_view){
#ifdef SUN3D
/*============================================================================*/
      float3 xyz = kfusion.pose.get_translation();
      cout << xyz.x << " " << xyz.y << " " << xyz.z << endl;
      float3 direction = kfusion.pose * make_float3(0, 0, 1) - xyz;
      double angle = atan2(direction.x, direction.z);
      cout << angle << endl;
      cout << direction.x << " " << direction.y << " " << direction.z <<  endl;
      rot = SE3<float>(makeVector(0.0, 0, 0, 0, angle, 0));

      renderInput( pos, normals, dep, kfusion.integration,
//          toMatrix4( rot * SE3<float>::exp(makeVector(xyz.x, xyz.y, xyz.z, 0, 0, 0)) ) * getInverseCameraMatrix(kfusion.configuration.camera * 2),
//          toMatrix4( SE3<float>(makeVector(size/2, size/2, 2, 0, 0, 0)) * rot ) * getInverseCameraMatrix(kfusion.configuration.camera * 2),
          toMatrix4( trans * rot * preTrans ) * getInverseCameraMatrix(kfusion.configuration.camera * 2),
          kfusion.configuration.nearPlane,
          kfusion.configuration.farPlane,
          kfusion.configuration.stepSize(),
          kfusion.configuration.mu * 0.75);
/*============================================================================*/
#else
      renderInput( pos, normals, dep, kfusion.integration, toMatrix4( trans * rot * preTrans ) * getInverseCameraMatrix(kfusion.configuration.camera * 2), kfusion.configuration.nearPlane, kfusion.configuration.farPlane, kfusion.configuration.stepSize(), 0.75 * kfusion.configuration.mu);
#endif
        count = 0;
        redraw_big_view = false;
    } else
        count++;
    if(render_texture)
        renderTexture( texModel.getDeviceImage(), pos, normals, rgbImage.getDeviceImage(), getCameraMatrix(2*kfusion.configuration.camera) * inverse(kfusion.pose), light);
    else
        renderLight( texModel.getDeviceImage(), pos, normals, light, ambient);
    cudaDeviceSynchronize();

    Stats.sample("render");

    glClear(GL_COLOR_BUFFER_BIT);
    glRasterPos2i(0, 0);
    glDrawPixels(lightScene);
    glRasterPos2i(0, 240);
    glPixelZoom(0.5, -0.5);
    glDrawPixels(rgbImage);
    glPixelZoom(1,-1);
    glRasterPos2i(320,0);
    glDrawPixels(lightModel);
    glRasterPos2i(320,240);
    glDrawPixels(trackModel);
    glRasterPos2i(640, 0);
    glDrawPixels(texModel);
    const double endProcessing = Stats.sample("draw");

    Stats.sample("total", endProcessing - startFrame, PerfStats::TIME);
    Stats.sample("total_proc", endProcessing - startProcessing, PerfStats::TIME);

    if(printCUDAError())
        exit(1);

    ++counter;

    if(counter % 50 == 0){
        Stats.print();
        Stats.reset();
        cout << endl;
    }

    glutSwapBuffers();
}

void idle(void){
#ifndef SUN3D
    if(KinectFrameAvailable())
#endif
        glutPostRedisplay();
}

void keys(unsigned char key, int x, int y){
    switch(key){
    case 'c':
        kfusion.Reset();
        kfusion.setPose(toMatrix4(initPose));
        reset = true;
        break;
    case 'q':
        exit(0);
        break;
    case 'i':
        should_integrate = !should_integrate;
        break;
    case 't':
        render_texture = !render_texture;
        break;
    }
}

void specials(int key, int x, int y){
    switch(key){
    case GLUT_KEY_LEFT:
        rot = SE3<float>(makeVector(0.0, 0, 0, 0, 0.1, 0)) * rot;
        break;
    case GLUT_KEY_RIGHT:
        rot = SE3<float>(makeVector(0.0, 0, 0, 0, -0.1, 0)) * rot;
        break;
    case GLUT_KEY_UP:
        rot *= SE3<float>(makeVector(0.0, 0, 0, -0.1, 0, 0));
        break;
    case GLUT_KEY_DOWN:
        rot *= SE3<float>(makeVector(0.0, 0, 0, 0.1, 0, 0));
        break;
    }
    redraw_big_view = true;
}

void reshape(int width, int height){
    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();
    glViewport(0, 0, width, height);
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    glColor3f(1.0f,1.0f,1.0f);
    glRasterPos2f(-1, 1);
    glOrtho(-0.375, width-0.375, height-0.375, -0.375, -1 , 1); //offsets to make (0,0) the top left pixel (rather than off the display)
    glPixelZoom(1,-1);
}

void exitFunc(void){
#ifndef SUN3D
    CloseKinect();
#endif
    kfusion.Clear();
    cudaDeviceReset();
}

int main(int argc, char ** argv) {
    size = (argc > 1) ? atof(argv[1]) : 8.f;

#ifdef SUN3D
/*============================================================================*/

    GetFileNames(image_dir, &image_list);
    GetFileNames(depth_dir, &depth_list);

    float fx, fy, cx, cy, ff;
    FILE *fp = fopen(intrinsic.c_str(), "r");
    fscanf(fp, "%f", &fx);
    fscanf(fp, "%f", &ff);
    fscanf(fp, "%f", &cx);
    fscanf(fp, "%f", &ff);
    fscanf(fp, "%f", &fy);
    fscanf(fp, "%f", &cy);

/*============================================================================*/
#endif

    KFusionConfig config;

    // it is enough now to set the volume resolution once.
    // everything else is derived from that.
    // config.volumeSize = make_uint3(64);
    // config.volumeSize = make_uint3(128);
//    config.volumeSize = make_uint3(256);
    config.volumeSize = make_uint3(512);

    // these are physical dimensions in meters
    config.volumeDimensions = make_float3(size);
    config.nearPlane = 0.4f;
    config.farPlane = 10.0f;
    config.mu = 0.1;
    config.combinedTrackAndReduce = false;

    // change the following parameters for using 640 x 480 input images
    config.inputSize = make_uint2(320,240);
#ifdef SUN3D
    config.camera =  make_float4(fx / 2.f, fy / 2.f, cx / 2.f, cy / 2.f);
#else
    config.camera =  make_float4(531.15/2, 531.15/2, 640/4, 480/4);
#endif
    // config.iterations is a vector<int>, the length determines
    // the number of levels to be used in tracking
    // push back more then 3 iteraton numbers to get more levels.
    config.iterations[0] = 10;
    config.iterations[1] = 5;
    config.iterations[2] = 4;

    config.dist_threshold = (argc > 2 ) ? atof(argv[2]) : config.dist_threshold;
    config.normal_threshold = (argc > 3 ) ? atof(argv[3]) : config.normal_threshold;

    initPose = SE3<float>(makeVector(size/2, size/2, 2, 0, 0, 0));

    glutInit(&argc, argv);
    glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE );
    glutInitWindowSize(config.inputSize.x * 2 + 640, max(config.inputSize.y * 2, 480));
    glutCreateWindow("kfusion");

    kfusion.Init(config);

    // input buffers
    depthImage[0].alloc(make_uint2(640, 480));
    depthImage[1].alloc(make_uint2(640, 480));
    rgbImage.alloc(make_uint2(640, 480));

    // render buffers
    lightScene.alloc(config.inputSize), trackModel.alloc(config.inputSize), lightModel.alloc(config.inputSize);
    pos.alloc(make_uint2(640, 480)), normals.alloc(make_uint2(640, 480)), dep.alloc(make_uint2(640, 480)), texModel.alloc(make_uint2(640, 480));

    if(printCUDAError()) {
        cudaDeviceReset();
        return 1;
    }

    memset(depthImage[0].data(), 0, depthImage[0].size.x*depthImage[0].size.y * sizeof(uint16_t));
    memset(depthImage[1].data(), 0, depthImage[1].size.x*depthImage[1].size.y * sizeof(uint16_t));
    memset(rgbImage.data(), 0, rgbImage.size.x*rgbImage.size.y * sizeof(uchar3));

#ifndef SUN3D
    uint16_t * buffers[2] = {depthImage[0].data(), depthImage[1].data()};
    if(InitKinect(buffers, (unsigned char *)rgbImage.data())){
        cudaDeviceReset();
        return 1;
    }
#endif

    kfusion.setPose(toMatrix4(initPose));

    // model rendering parameters
#ifdef SUN3D
    preTrans = SE3<float>::exp(makeVector(0.0, 0, -size, 0, 0, 0));
    trans = SE3<float>::exp(makeVector(0.5, 0.5, 0.5, 0, 0, 0) * size);
#else
    preTrans = SE3<float>::exp(makeVector(0.0, 0, -size, 0, 0, 0));
    trans = SE3<float>::exp(makeVector(0.5, 0.5, 0.5, 0, 0, 0) * size);
#endif

    atexit(exitFunc);
    glutDisplayFunc(display);
    glutKeyboardFunc(keys);
    glutSpecialFunc(specials);
    glutReshapeFunc(reshape);
    glutIdleFunc(idle);

    glutMainLoop();

#ifndef SUN3D
    CloseKinect();
#endif

    return 0;
}
