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
#include <fstream>
#include <sstream>
#include <iomanip>
#include <cstring>
#include <dirent.h>
#include <cerrno>
#include <cmath>

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

/*============================================================================*/
#define SUN3D
//#define RENDER_SCENE
//#define LOCAL_RUN
#define INITIAL_POSE
/*============================================================================*/

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
int integration_rate = 1;
bool reset = true;
bool should_integrate = true;
bool render_texture = false;

Image<float3, Device> pos, normals;
Image<float, Device> dep;
SE3<float> preTrans, trans, rot(makeVector(0.0, 0, 0, 0, 0, 0));
bool redraw_big_view = false;

/*============================================================================*/
#ifdef SUN3D

Image<uint16_t, HostDevice> fusedDepth;

////////////////////////////////////////////////////////////////////////////////
// global parameter

int   param_start_index = 1866;

int   param_volume_size = 640;			// 715 is maximum
float param_volume_dimension = 8.f;

int   param_frame_threshold = 200;
float param_angle_factor = 1.f;
float param_translation_factor = 1.f;
float param_rsme_threshold = 1.5e-2f;

int   param_file_name_length = 24;

enum  KinfuMode {KINFU_FORWARD, KINFU_BACKWARD};
KinfuMode param_mode = KINFU_FORWARD;

// voxel resolution: 0.01 meter

////////////////////////////////////////////////////////////////////////////////

int file_index;
float angle_threshold, translation_threshold;

const int kImageRows = 480;
const int kImageCols = 640;
const int kImageChannels = 3;

vector<string> image_list;
vector<string> depth_list;
vector<string> extrinsic_list;

#ifdef INITIAL_POSE
vector<Matrix4> extrinsic_poses;
map<int, Matrix4> pose_map;
#endif

string image_dir, depth_dir, fused_dir, extrinsic_dir, frame_dir, pose_dir;

////////////////////////////////////////////////////////////////////////////////

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

////////////////////////////////////////////////////////////////////////////////

bool GetDepthData(string file_name, uint16_t *data) {
  png::image< png::gray_pixel_16 > img(file_name.c_str(),
      png::require_color_space< png::gray_pixel_16 >());

  int index = 0;
  for (int i = 0; i < kImageRows; ++i) {
    for (int j = 0; j < kImageCols; ++j) {
      uint16_t s = img.get_pixel(j, i);
      *(data + index) = (s << 13 | s >> 3);
      ++index;
    }
  }

  return true;
}

////////////////////////////////////////////////////////////////////////////////

void SaveFusedDepthFile() {
	string depth_full_name = depth_list[param_start_index];
	string depth_serial_name = depth_full_name.substr(
			depth_full_name.size() - param_file_name_length, param_file_name_length);
	string fused_full_name = fused_dir + depth_serial_name;

#ifdef RESOLUTION_1280X960
	png::image<png::gray_pixel_16> img(kImageCols * 2, kImageRows * 2);

	kfusion.Raycast_2();
	renderFusedMap(fusedDepth.getDeviceImage(), kfusion.vertex_2, inverse(kfusion.pose));

	for (int i = 0; i < kImageRows * 2; ++i) {
		for (int j = 0; j < kImageCols * 2; ++j) {
			uint16_t s = fusedDepth[make_uint2(j,i)];
			img[i][j] = (s >> 13 | s << 3);
		}
	}
#else
	png::image<png::gray_pixel_16> img(kImageCols, kImageRows);

	renderFusedMap(fusedDepth.getDeviceImage(), kfusion.vertex, inverse(kfusion.pose));

	for (int i = 0; i < kImageRows; ++i) {
		for (int j = 0; j < kImageCols; ++j) {
			uint16_t s = fusedDepth[make_uint2(j,i)];
			img[i][j] = (s >> 13 | s << 3);
		}
	}
#endif

	img.write(fused_full_name.c_str());

#ifdef INITIAL_POSE
#ifndef RESOLUTION_1280X960
	string serial_txt = depth_serial_name.substr(0, param_file_name_length - 4) + ".txt";
	string pose_txt_name  = pose_dir  + serial_txt;
	string frame_txt_name = frame_dir + serial_txt;

	FILE *fp_frame = fopen(frame_txt_name.c_str(), "w");
	ofstream pose_file;
	pose_file.open(pose_txt_name.c_str());
	pose_file.precision(60);

	for (map<int, Matrix4>::iterator it = pose_map.begin(); it != pose_map.end(); ++it) {
		fprintf(fp_frame, "%d\n", it->first);
		Matrix4 m = it->second;
		for (int i = 0; i < 3; ++i) {
			pose_file << m.data[i].x << "\t";
			pose_file << m.data[i].y << "\t";
			pose_file << m.data[i].z << "\t";
			pose_file << m.data[i].w << "\n";
		}
	}
	fclose(fp_frame);
	pose_file.close();
#endif
#endif
}

////////////////////////////////////////////////////////////////////////////////

bool GetExtrinsicData(string file_name, vector<Matrix4> *poses) {
	FILE *fp = fopen(file_name.c_str(), "r");
	for (int i = 0; i < image_list.size(); ++i) {
		Matrix4 m;
		for (int d = 0; d < 3; ++d) {
			int iret;
			iret = fscanf(fp, "%f", &m.data[d].x);
			iret = fscanf(fp, "%f", &m.data[d].y);
			iret = fscanf(fp, "%f", &m.data[d].z);
			iret = fscanf(fp, "%f", &m.data[d].w);
		}
		m.data[3].x = m.data[3].y = m.data[3].z = 0.f;
		m.data[3].w = 1.f;
		poses->push_back(m);
	}

	return true;
}

////////////////////////////////////////////////////////////////////////////////

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

/*----------------------------------------------------------------------------*/

void display(void){
    static bool integrate = true;

/*============================================================================*/

#ifdef SUN3D

    if (param_mode == KINFU_FORWARD) {
    	if (file_index == param_start_index + param_frame_threshold ||
    			file_index == image_list.size()) {
            kfusion.Integrate();

            param_mode = KINFU_BACKWARD;
            file_index = param_start_index - 1;
            kfusion.setPose(toMatrix4(initPose));

            kfusion.Raycast();

    		cout << "IDX" << endl << endl;
            return;
    	}

#ifdef INITIAL_POSE
    	// T_12 = T_01^(-1) * T_02
    	// T_02 = T_01 * T_12;
    	if (file_index > 0 && file_index != param_start_index) {
    		Matrix4 delta = inverse(extrinsic_poses[file_index - 1]) * extrinsic_poses[file_index];
    		kfusion.pose = kfusion.pose * delta;
    	}
#endif
    } else {
    	if (file_index == param_start_index - param_frame_threshold ||
    			file_index == -1) {
    		kfusion.Integrate();

    		kfusion.setPose(toMatrix4(initPose));

    		kfusion.Raycast();
    		SaveFusedDepthFile();

            cout << "IDX" << endl << endl;
    		exit(0);
    	}

#ifdef INITIAL_POSE
		Matrix4 delta = inverse(extrinsic_poses[file_index + 1]) * extrinsic_poses[file_index];
		kfusion.pose = kfusion.pose * delta;
#endif
    }

    cout << file_index << " ";
    cout.flush();

#ifdef RENDER_SCENE
    GetImageData(image_list[file_index], (unsigned char *)rgbImage.data());
#endif
    GetDepthData(depth_list[file_index], (uint16_t *)depthImage[0].data());

    glClear( GL_COLOR_BUFFER_BIT );
    const double startFrame = Stats.start();
    const double startProcessing = Stats.sample("kinect");

    kfusion.setKinectDeviceDepth(depthImage[0].getDeviceImage());
    Stats.sample("raw to cooked");

/*----------------------------------------------------------------------------*/
#else
    glClear( GL_COLOR_BUFFER_BIT );
    const double startFrame = Stats.start();
    const double startProcessing = Stats.sample("kinect");

    kfusion.setKinectDeviceDepth(depthImage[GetKinectFrame()].getDeviceImage());
    Stats.sample("raw to cooked");
#endif

#if 0
    kfusion.Integrate();
	kfusion.Raycast();
	SaveFusedDepthFile();
	exit(0);
#endif

    integrate = kfusion.Track();
    Stats.sample("track");

#ifdef SUN3D
/*============================================================================*/

#ifdef INITIAL_POSE
    pose_map.insert(make_pair(file_index, kfusion.pose));
#endif

    double z_angle;
    Vector<3, float> diff_t;
    diff_t[0] = diff_t[1] = diff_t[2] = 0.f;

    if (file_index != param_start_index) {
    	float3 cam_z;
    	cam_z.x = cam_z.y = 0.f;
    	cam_z.z = 1.f;
    	float3 wor_z = kfusion.pose * cam_z;
    	z_angle = acos(wor_z.z);

		float3 temp_t = kfusion.pose.get_translation();
		Vector<3, float> curr_t;
		curr_t[0] = temp_t.x;
		curr_t[1] = temp_t.y;
		curr_t[2] = temp_t.z;
		Vector<3, float> init_t = initPose.get_translation();
		diff_t = curr_t - init_t;
    }

    if ((!integrate && file_index != param_start_index) ||
    		z_angle > angle_threshold * param_angle_factor ||
    		norm(diff_t) > translation_threshold * param_translation_factor ) {
		if (param_mode == KINFU_FORWARD) {
			kfusion.Integrate();

			param_mode = KINFU_BACKWARD;
			file_index = param_start_index - 1;
			kfusion.setPose(toMatrix4(initPose));

			kfusion.Raycast();

			cout << "THR" << endl << endl;
			return;
		} else {
			kfusion.Integrate();

			kfusion.setPose(toMatrix4(initPose));

			kfusion.Raycast();
			SaveFusedDepthFile();

            cout << "THR" << endl << endl;
			exit(0);
		}
    }

    if (param_mode == KINFU_FORWARD)
    	++file_index;
    else
    	--file_index;
/*----------------------------------------------------------------------------*/
#endif

    if((should_integrate && integrate && ((counter % integration_rate) == 0)) || reset){
        kfusion.Integrate();
        kfusion.Raycast();
        Stats.sample("integrate");
        if(counter > 2) // use the first two frames to initialize
            reset = false;
    }

#ifdef RENDER_SCENE
	renderLight( lightScene.getDeviceImage(), kfusion.inputVertex[0], kfusion.inputNormal[0], light, ambient );
    renderLight( lightModel.getDeviceImage(), kfusion.vertex, kfusion.normal, light, ambient);
    renderTrackResult(trackModel.getDeviceImage(), kfusion.reduction);
    static int count = 4;
    if(count > 3 || redraw_big_view){
#ifdef SUN3D
/*============================================================================*/
      float3 xyz = kfusion.pose.get_translation();
      float3 direction = kfusion.pose * make_float3(0, 0, 1) - xyz;
      double angle = atan2(direction.x, direction.z);
      rot = SE3<float>(makeVector(0, 0, 0, 0, angle, 0));

      renderInput( pos, normals, dep, kfusion.integration,
          toMatrix4( trans * rot * preTrans ) * getInverseCameraMatrix(kfusion.configuration.camera),
          kfusion.configuration.nearPlane,
          kfusion.configuration.farPlane,
          kfusion.configuration.stepSize(),
          kfusion.configuration.mu * 0.75);
/*----------------------------------------------------------------------------*/
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
#endif

    cudaDeviceSynchronize();

    Stats.sample("render");

#ifdef RENDER_SCENE
    glClear(GL_COLOR_BUFFER_BIT);
    glRasterPos2i(0, 0);
    glDrawPixels(lightScene);
    glRasterPos2i(0, kImageRows);
    glPixelZoom(1, -1);
    glDrawPixels(rgbImage);
    glRasterPos2i(kImageCols,0);
    glDrawPixels(lightModel);
    glRasterPos2i(kImageCols, kImageRows);
    glDrawPixels(trackModel);
    glRasterPos2i(kImageCols * 2, 0);
    glPixelZoom(2, -2);
    glDrawPixels(texModel);
#endif

    const double endProcessing = Stats.sample("draw");

    Stats.sample("total", endProcessing - startFrame, PerfStats::TIME);
    Stats.sample("total_proc", endProcessing - startProcessing, PerfStats::TIME);

    if(printCUDAError())
        exit(1);

    ++counter;

//    if(counter % 50 == 0){
//        Stats.print();
//        Stats.reset();
//        cout << endl;
//    }

#ifdef RENDER_SCENE
    glutSwapBuffers();
#endif

//    usleep(1000 * 500);
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
#ifdef SUN3D
/*============================================================================*/

	cout << "=================================================================" << endl;

	string data_prefix, sfm_prefix, data_dir, sfm_dir, data_name;

	switch (argc) {
	case 1:
		data_prefix = "/home/alan/DATA/SUN3D/";
		sfm_prefix  = data_dir;
		data_name   = "hotel_umd/maryland_hotel3/";
		break;
	case 4:
		data_prefix = argv[1];
		sfm_prefix  = argv[2];
		data_name   = argv[3];

		break;
	case 5:
		break;
	default:
		cout << "Wrong arguments...";
		exit(0);
	}

	data_dir = data_prefix + data_name;
	sfm_dir  = sfm_prefix  + data_name;
	image_dir = data_dir + "image/";
	depth_dir = data_dir + "depth/";
	extrinsic_dir = data_dir + "extrinsics/";
	frame_dir = sfm_dir + "frameTSDF/";
	pose_dir  = sfm_dir + "poseTSDF/";

#ifdef RESOLUTION_1280X960
	fused_dir = data_dir + "depth1280x960/";
#else
	fused_dir = data_dir + "depthTSDF/";
#endif

	file_index = param_start_index;

    size = param_volume_dimension;

    GetFileNames(image_dir, &image_list);
    GetFileNames(depth_dir, &depth_list);
    GetFileNames(extrinsic_dir, &extrinsic_list);
#ifdef INITIAL_POSE
    GetExtrinsicData(extrinsic_list[extrinsic_list.size() - 1], &extrinsic_poses);
#endif

    int i_ret;
    float fx, fy, cx, cy, ff;
	string intrinsic = data_dir + "intrinsics.txt";
    FILE *fp = fopen(intrinsic.c_str(), "r");
    i_ret = fscanf(fp, "%f", &fx);
    i_ret = fscanf(fp, "%f", &ff);
    i_ret = fscanf(fp, "%f", &cx);
    i_ret = fscanf(fp, "%f", &ff);
    i_ret = fscanf(fp, "%f", &fy);
    i_ret = fscanf(fp, "%f", &cy);

    angle_threshold = (float) atan(cy / fy);
    translation_threshold = 1.0f * cy / fy;

/*----------------------------------------------------------------------------*/
#endif

    KFusionConfig config;

    // it is enough now to set the volume resolution once.
    // everything else is derived from that.
    // config.volumeSize = make_uint3(64);
    // config.volumeSize = make_uint3(128);
    // config.volumeSize = make_uint3(256);
    config.volumeSize = make_uint3(param_volume_size);

    // these are physical dimensions in meters
    config.volumeDimensions = make_float3(size);
    config.nearPlane = 0.4f;
    config.farPlane = 10.0f;
    config.mu = 0.1;
    config.combinedTrackAndReduce = false;

    // change the following parameters for using 640 x 480 input images
    uint2 input_size = make_uint2(kImageCols, kImageRows);
    config.inputSize = input_size;

#ifdef SUN3D
    config.camera =  make_float4(fx, fy, cx, cy);

    config.rsme_threshold = param_rsme_threshold;
//    config.track_threshold = 0.7f;
#else
    config.camera =  make_float4(531.15/2, 531.15/2, 640/4, 480/4);
#endif
    // config.iterations is a vector<int>, the length determines
    // the number of levels to be used in tracking
    // push back more then 3 iteraton numbers to get more levels.
    config.iterations[0] = 10;
    config.iterations[1] = 5;
    config.iterations[2] = 4;

    initPose = SE3<float>(makeVector(size/2, size/2, 0, 0, 0, 0));

#ifdef RENDER_SCENE
    glutInit(&argc, argv);
    glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE );
    glutInitWindowSize(config.inputSize.x * 2 + kImageCols * 2, max(config.inputSize.y * 2, kImageRows * 2));
    glutCreateWindow("kfusion");
#endif

    kfusion.Init(config);

    // input buffers
    depthImage[0].alloc(input_size);
    depthImage[1].alloc(input_size);
    rgbImage.alloc(input_size);

    // render buffers
    lightScene.alloc(input_size), trackModel.alloc(input_size), lightModel.alloc(input_size);
    pos.alloc(input_size), normals.alloc(input_size), dep.alloc(input_size), texModel.alloc(input_size * 2);

    if(printCUDAError()) {
        cudaDeviceReset();
        return 1;
    }

    memset(depthImage[0].data(), 0, depthImage[0].size.x*depthImage[0].size.y * sizeof(uint16_t));
    memset(depthImage[1].data(), 0, depthImage[1].size.x*depthImage[1].size.y * sizeof(uint16_t));
    memset(rgbImage.data(), 0, rgbImage.size.x*rgbImage.size.y * sizeof(uchar3));

#ifdef SUN3D

#ifdef RESOLUTION_1280X960
    fusedDepth.alloc(input_size * 2);
#else
    fusedDepth.alloc(input_size);
#endif

#else
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

#ifdef RENDER_SCENE
    atexit(exitFunc);
    glutDisplayFunc(display);
    glutKeyboardFunc(keys);
    glutSpecialFunc(specials);
    glutReshapeFunc(reshape);
    glutIdleFunc(idle);

    glutMainLoop();
#else

    while(1)
    	display();

#endif

#ifndef SUN3D
    CloseKinect();
#endif

    return 0;
}
