//cmake -DCMAKE_BUILD_TYPE=Debug -DOpenGL_GL_PREFERENCE=GLVND ..

/**
 *   #, #,         CCCCCC  VV    VV MM      MM RRRRRRR
 *  %  %(  #%%#   CC    CC VV    VV MMM    MMM RR    RR
 *  %    %## #    CC        V    V  MM M  M MM RR    RR
 *   ,%      %    CC        VV  VV  MM  MM  MM RRRRRR
 *   (%      %,   CC    CC   VVVV   MM      MM RR   RR
 *     #%    %*    CCCCCC     VV    MM      MM RR    RR
 *    .%    %/
 *       (%.      Computer Vision & Mixed Reality Group
 *                For more information see <http://cvmr.info>
 *
 * This file is part of RBOT.
 *
 *  @copyright:   RheinMain University of Applied Sciences
 *                Wiesbaden Rüsselsheim
 *                Germany
 *     @author:   Henning Tjaden
 *                <henning dot tjaden at gmail dot com>
 *    @version:   1.0
 *       @date:   30.08.2018
 *
 * RBOT is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * RBOT is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with RBOT. If not, see <http://www.gnu.org/licenses/>.
 */

#include <QApplication>
#include <QSettings>
#include <QThread>

#include <chrono>
#include <fstream>
#include <iomanip>
#include <thread>
#include <experimental/filesystem>
#include <unistd.h>
//#include <sys/types.h>
#include <sys/inotify.h>

#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>

#include "object3d.h"
#include "pose_estimator6d.h"

using namespace std;
using namespace cv;

void LogError(const cv::Matx44f& M, const cv::Matx44f& gt_M, const size_t i) {
  auto err_R = std::numeric_limits<float>::quiet_NaN();
  auto err_t = std::numeric_limits<float>::quiet_NaN();
  const auto R = M.get_minor<3, 3>(0, 0);
  const auto gt_R = gt_M.get_minor<3, 3>(0, 0);
  if (cv::norm(R) > 0.1) {
    err_R = std::abs(std::acos(0.5f * cv::trace(R * gt_R.t()) - 0.5f));
    err_t = std::sqrt(std::pow(M(0,3) - gt_M(0,3), 2.0f) + std::pow(M(1,3) - gt_M(1,3), 2.0f) + std::pow(M(2,3) - gt_M(2,3), 2.0f));
  }
  std::cout << std::setfill('0') << std::setw(4) << i << " " << err_R  << " " << err_t << std::endl;
}

char blend(const unsigned char i1, const unsigned char i2, const float alpha) {
  return static_cast<unsigned char>(std::min(255.0f, alpha * static_cast<float>(i1) + (1 - alpha) * static_cast<float>(i2)));
}

cv::Mat drawResultOverlay(const vector<Object3D*>& objects, const cv::Mat& frame)
{
    // render the models with phong shading
    RenderingEngine::Instance()->setLevel(0);

    vector<Point3f> colors;
    //colors.push_back(Point3f(1.0, 0.5, 0.0));
    colors.push_back(Point3f(0.2, 0.3, 1.0));
    RenderingEngine::Instance()->renderShaded(vector<Model*>(objects.begin(), objects.end()), GL_LINES, colors, true);

    // download the rendering to the CPU
    Mat rendering = RenderingEngine::Instance()->downloadFrame(RenderingEngine::RGB);

    // download the depth buffer to the CPU
    Mat depth = RenderingEngine::Instance()->downloadFrame(RenderingEngine::DEPTH);

    // compose the rendering with the current camera image for demo purposes (can be done more efficiently directly in OpenGL)
    Mat result = frame.clone();
    const float alpha = 0.65f;
    for(int y = 0; y < frame.rows; y++)
    {
        for(int x = 0; x < frame.cols; x++)
        {
            Vec3b color = rendering.at<Vec3b>(y,x);
            if(depth.at<float>(y,x) != 0.0f)
            {
                result.at<Vec3b>(y,x)[0] = blend(result.at<Vec3b>(y,x)[0], color[2], alpha);
                result.at<Vec3b>(y,x)[1] = blend(result.at<Vec3b>(y,x)[1], color[1], alpha);
                result.at<Vec3b>(y,x)[2] = blend(result.at<Vec3b>(y,x)[2], color[0], alpha);
            }
        }
    }
    return result;
}

class Inotify {
 public:
  Inotify(const std::string& target, const std::string& pattern)
    : target_(target),
      pattern_(pattern) {}

  ~Inotify() {
    if (fd_ > 0) {
      if (wd_ > 0) {
        (void)inotify_rm_watch(fd_, wd_);
      }
      (void)close(fd_);
    }
  }

  int Init() {
    fd_ = inotify_init();
    if (fd_ < 0) {
      return -1;
    }

    wd_ = inotify_add_watch(fd_, target_.c_str(), mask_);
    if (wd_ < 0) {
      (void)close(fd_);
      return -1;
    }

    return 0;
  }

  int GetEvents() {
    index_ = 0;
    bytes_ = read(fd_, buffer_, sizeof(buffer_));
    return (bytes_ < 0)? -1 : 0;
  }

  std::string GetFrame() {
    if (index_ >= bytes_) {
      if (GetEvents() < 0) {
        return std::string();
      }
    }

    while(index_ < bytes_) {
      struct inotify_event *event = (struct inotify_event*)(&buffer_[index_]);
      index_ += sizeof(struct inotify_event) + event->len;

      if (event->len && (event->mask & mask_) && !(event->mask & IN_ISDIR)) {
        std::string frame_path = target_ + "/" + std::string(event->name);
        if (std::string::npos != frame_path.find(pattern_)) {
          return std::move(frame_path);
        }
      }
    }

    return std::string();
  }

 private:
  int fd_{-1};
  int wd_{-1};
  int index_{-1};
  int bytes_{-1};
  uint32_t mask_{IN_CLOSE_WRITE};
  char buffer_[1024 * (sizeof(struct inotify_event) + 16)];
  std::string target_;
  std::string pattern_;
};

int main(int argc, char *argv[])
{
    QApplication a(argc, argv);

    if (argc != 7 && argc != 8 && argc != 9) {
      std::cerr << "usage: " << argv[0] << " shader_folder dataset_folder object_name nbins iterations mode [live_frames_folder] [live_frames_pattern]" << std::endl;
      return 1;
    }
    const std::string shader_folder = std::string(argv[1]);
    assert(std::experimental::filesystem::exists(shader_folder));
    const std::string dataset_folder = std::string(argv[2]);
    assert(std::experimental::filesystem::exists(dataset_folder));
    const std::string object_name = std::string(argv[3]);
    const int nbins = std::atoi(argv[4]);
    const size_t iterations = std::atoi(argv[5]);
    constexpr int MODE_DEMO = 0;
    constexpr int MODE_DEMO_AUTO = 1;
    constexpr int MODE_TESTING = 2;
    constexpr int MODE_POSE_FINDER = 3;
    constexpr int MODE_POSE_FINDER_LIVE = 4;
    const int mode = std::atoi(argv[6]);
    const std::string live_frames_folder = std::string((argc >= 8)? argv[7] : "");
    assert(std::experimental::filesystem::exists(live_frames_folder));
    const std::string live_frames_pattern = std::string((argc == 9)? argv[8] : "");
/*
    const std::string pipe_path(argv[7]);
    std::ofstream pipe(pipe_path);
    pipe << getpid() << std::endl;
    pipe.close();
    std::this_thread::sleep_for(std::chrono::milliseconds(500));
*/
    // frames paths
    std::vector<std::string> images;
    const std::string frames_folder = dataset_folder + "/" + object_name + "/frames";
    assert(std::experimental::filesystem::exists(frames_folder));
    for (const auto & entry : std::experimental::filesystem::directory_iterator(frames_folder)) {
      const auto path = entry.path().string();
      if (std::string::npos == path.find("a_")) {
        continue;
      }
      images.push_back(path);
    }
    if (std::string::npos != object_name.find("gripper")) {
      std::sort(images.begin(), images.end(), std::greater<std::string>());
    } else {
      std::sort(images.begin(), images.end());
    }

    // load and validate info
    /*{
      std::string line;
      std::ifstream inifile(dataset_folder + "/" + object_name + "/" + object_name + ".ini");
      if (inifile.is_open()) {
        while (getline(inifile, line)) {
          std::cout << line << std::endl;
        }
      }
      inifile.close();
    }*/
    const QString info_path = QString::fromStdString(dataset_folder + "/" + object_name + "/" + object_name + ".ini");
    QSettings info(info_path, QSettings::IniFormat);
    // -- input frames range (optional)
    int frame_first = info.value("frames/first", 0).toInt();
    int frame_last = info.value("frames/last", (static_cast<int>(images.size()) - 1)).toInt();
    assert(frame_first >= 0 && frame_first < images.size());
    assert(frame_last > frame_first && frame_last < images.size());
    // -- image scaling (optional)
    float scale = info.value("image/scale", 1.0f).toFloat();
    assert(scale > 0.0f && scale <= 1.0f);
    // -- image dimensions
    int width = info.value("image/width", std::numeric_limits<int>::max()).toInt();
    int height = info.value("image/height", std::numeric_limits<int>::max()).toInt();
    assert(width != std::numeric_limits<int>::max());
    assert(height != std::numeric_limits<int>::max());
    width = static_cast<int>(static_cast<float>(width) * scale);
    height = static_cast<int>(static_cast<float>(height) * scale);
    // -- near and far plane of the OpenGL view frustum
    float zNear = info.value("frustum/znear", std::numeric_limits<float>::max()).toFloat();
    float zFar = info.value("frustum/zfar", std::numeric_limits<float>::max()).toFloat();
    assert(zNear != std::numeric_limits<float>::max());
    assert(zFar != std::numeric_limits<float>::max());
    // -- camera intrinsics
    float fx = info.value("K/fx", std::numeric_limits<float>::max()).toFloat();
    float fy = info.value("K/fy", std::numeric_limits<float>::max()).toFloat();
    float cx = info.value("K/cx", std::numeric_limits<float>::max()).toFloat();
    float cy = info.value("K/cy", std::numeric_limits<float>::max()).toFloat();
    assert(fx != std::numeric_limits<float>::max()); fx *= scale;
    assert(fy != std::numeric_limits<float>::max()); fy *= scale;
    assert(cx != std::numeric_limits<float>::max()); cx *= scale;
    assert(cy != std::numeric_limits<float>::max()); cy *= scale;
    Matx33f K = Matx33f(fx, 0, cx, 0, fy, cy, 0, 0, 1);
    // -- radial distortion
    float d0 = info.value("distortion/d0", std::numeric_limits<float>::max()).toFloat();
    float d1 = info.value("distortion/d1", std::numeric_limits<float>::max()).toFloat();
    float d2 = info.value("distortion/d2", std::numeric_limits<float>::max()).toFloat();
    float d3 = info.value("distortion/d3", std::numeric_limits<float>::max()).toFloat();
    //float d4 = info.value("distortion/d4", std::numeric_limits<float>::max()).toFloat();
    assert(d0 != std::numeric_limits<float>::max());
    assert(d1 != std::numeric_limits<float>::max());
    assert(d2 != std::numeric_limits<float>::max());
    assert(d3 != std::numeric_limits<float>::max());
    //assert(d4 != std::numeric_limits<float>::max());
    Matx14f distCoeffs =  Matx14f(d0, d1, d2, d3);
    // -- initial pose
    float x = info.value("pose/x", std::numeric_limits<float>::max()).toFloat();
    float y = info.value("pose/y", std::numeric_limits<float>::max()).toFloat();
    float z = info.value("pose/z", std::numeric_limits<float>::max()).toFloat();
    float roll = info.value("pose/roll", std::numeric_limits<float>::max()).toFloat();
    float pitch = info.value("pose/pitch", std::numeric_limits<float>::max()).toFloat();
    float yaw = info.value("pose/yaw", std::numeric_limits<float>::max()).toFloat();
    assert(x != std::numeric_limits<float>::max());
    assert(y != std::numeric_limits<float>::max());
    assert(z != std::numeric_limits<float>::max());
    assert(roll != std::numeric_limits<float>::max());
    assert(pitch != std::numeric_limits<float>::max());
    assert(yaw != std::numeric_limits<float>::max());

    std::cerr << "\n         image scale: " << scale
              << "\n          image size: {w:" << width << ", h:" << height << "}"
              << "\n        view frustum: {znear:" << zNear << ", zfar:" << zFar << "}"
              << "\n         calibration: {fx:" << fx << ", fy:" << fy << ", cx:" << cx << ", cy:" << cy << "}"
              << "\n          distortion: {d0:" << d0 << ", d1:" << d1 << ", d2:" << d2 << ", d3:" << d3 << "}"
              << "\n    initial position: {x:" << x << ", y:" << y << ", z:" << z <<  "}"
              << "\n initial orientation: {roll:" << roll << ", pitch:" << pitch << ", yaw:" << yaw << "}"
              << "\n       shader folder: " << std::quoted(shader_folder)
              << "\n      dataset folder: " << std::quoted(dataset_folder)
              << "\n         object name: " << std::quoted(object_name)
              << "\n         first frame: " << frame_first
              << "\n          last frame: " << frame_last
              << "\n      number of bins: " << nbins
              << "\n          iterations: " << iterations
              << "\n          frames dir: " << std::quoted(live_frames_folder)
              << "\n      frames pattern: " << std::quoted(live_frames_pattern)
              << "\n                mode: " << mode;

    // distances for the pose detection template generation
    std::vector<float> distances = {200.0f, 400.0f, 600.0f};

    // load 3D objects
    std::vector<Object3D*> objects;
    const std::string mesh_path = dataset_folder + "/" + object_name  + "/" + object_name + ".obj";
    objects.push_back(new Object3D(mesh_path, (mode == MODE_POSE_FINDER || mode == MODE_POSE_FINDER_LIVE)? 4 : nbins, x, y, z, roll, pitch, yaw, 1.0, 0.55f, distances));
    //objects.push_back(new Object3D("data/a_second_model.obj", -50, 0, 600, 30, 0, 180, 1.0, 0.55f, distances2));

    // create the pose estimator
    PoseEstimator6D* poseEstimator = new PoseEstimator6D(shader_folder, width, height, zNear, zFar, K, distCoeffs, objects);

    // move the OpenGL context for offscreen rendering to the current thread, if run in a seperate QT worker thread (unnessary in this example)
    //RenderingEngine::Instance()->getContext()->moveToThread(this);

    // active the OpenGL context for the offscreen rendering engine during pose estimation
    RenderingEngine::Instance()->makeCurrent();

    if (mode == MODE_POSE_FINDER || mode == MODE_POSE_FINDER_LIVE) {
      Mat frame;
      bool all_ok = true;
      Inotify inotify(live_frames_folder, live_frames_pattern);
      if (mode == MODE_POSE_FINDER) {
        frame = cv::imread(images[frame_first]);
        if (frame.empty()) {
          std::cerr << "cannot load frame " << images[frame_first] << std::endl;
          all_ok = false;
        } else {
          if (scale != 1.0f) {
            cv::resize(frame, frame, cv::Size(width, height));
          }
        }
      } else {
        if (inotify.Init() < 0) {
          std::cerr << "cannot monitor events at " << live_frames_folder << std::endl;
          all_ok = false;
        }
      }
      if (all_ok) {
        constexpr int HALF_RANGE = 1000;
        int roll_slider = HALF_RANGE, pitch_slider = HALF_RANGE, yaw_slider = HALF_RANGE;
        int x_slider = HALF_RANGE, y_slider = HALF_RANGE, z_slider = HALF_RANGE;
        cv::namedWindow("result", WINDOW_AUTOSIZE);
        cv::createTrackbar("roll", "result", &roll_slider, 2 * HALF_RANGE);
        cv::createTrackbar("pitch", "result", &pitch_slider, 2 * HALF_RANGE);
        cv::createTrackbar("yaw", "result", &yaw_slider, 2 * HALF_RANGE);
        cv::createTrackbar("x", "result", &x_slider, 2 * HALF_RANGE);
        cv::createTrackbar("y", "result", &y_slider, 2 * HALF_RANGE);
        cv::createTrackbar("z", "result", &z_slider, 2 * HALF_RANGE);
        while (0x1b != cv::waitKey(1)) {
          if (mode == MODE_POSE_FINDER_LIVE) {
            std::string frame_path = inotify.GetFrame();
            if (frame_path.empty()) {
              std::cerr << "empty frame path, stopping pose loop" << std::endl;
              break;
            }
            frame = cv::imread(frame_path);
            std::experimental::filesystem::remove(frame_path);
            if (frame.empty()) {
              std::cerr << "invalid frame path " << std::quoted(frame_path) << std::endl;
              continue;
            }
            if (scale != 1.0f) {
              cv::resize(frame, frame, cv::Size(width, height));
            }
          }
          const float roll = - 180.0f + static_cast<float>(roll_slider) * 180.0f / static_cast<float>(HALF_RANGE);
          const float pitch = - 180.0f + static_cast<float>(pitch_slider) * 180.0f / static_cast<float>(HALF_RANGE);
          const float yaw = - 180.0f + static_cast<float>(yaw_slider) * 180.0f / static_cast<float>(HALF_RANGE);
          const float x = static_cast<float>(- HALF_RANGE + x_slider);
          const float y = static_cast<float>(- HALF_RANGE + y_slider);
          const float z = static_cast<float>(- HALF_RANGE + z_slider);
          std::cerr << "x=" << x << "\ny=" << y << "\nz=" << z << "\nroll=" << roll << "\npitch=" << pitch << "\nyaw=" << yaw << std::endl << std::endl;
          cv::Matx44f pose = Transformations::rotationMatrix(roll, cv::Vec3f(1.0f, 0.0f, 0.0f)) *
                             Transformations::rotationMatrix(pitch, cv::Vec3f(0.0f, 1.0f, 0.0f)) *
                             Transformations::rotationMatrix(yaw, cv::Vec3f(0.0f, 0.0f, 1.0f));
          pose(0,3) = x; pose(1,3) = y; pose(2,3) = z;
          objects[0]->setPose(pose);

          Mat result = drawResultOverlay(objects, frame);
          cv::imshow("result", result);
        }
      }
    } else {
      std::vector<cv::Matx44f> ground_truth;
      if (mode == MODE_TESTING) {
        std::string gt_line;
        std::ifstream gt_stream;
        gt_stream.open(dataset_folder + "/" + "poses_first.txt");
        std::getline(gt_stream, gt_line); // ignore first line
        while (std::getline(gt_stream, gt_line)) {
          cv::Matx44f gt;
          std::istringstream ss(gt_line);
          ss >> gt(0, 0) >> gt(0, 1) >> gt(0, 2); gt(3, 0) = 0.0f;
          ss >> gt(1, 0) >> gt(1, 1) >> gt(1, 2); gt(3, 1) = 0.0f;
          ss >> gt(2, 0) >> gt(2, 1) >> gt(2, 2); gt(3, 2) = 0.0f;
          ss >> gt(0, 3) >> gt(1, 3) >> gt(2, 3); gt(3, 3) = 1.0f;
          ground_truth.push_back(gt);
        }
        gt_stream.close();

        assert(images.size() == ground_truth.size());

        Mat frame = cv::imread(images[frame_first]);
        if (frame.empty()) {
          std::cerr << "cannot load frame " << images[frame_first] << std::endl;
          return 1;
        }
        if (scale != 1.0f) {
          cv::resize(frame, frame, cv::Size(width, height));
        }
        objects[0]->setPose(ground_truth[frame_first]);
        poseEstimator->toggleTracking(frame, 0, false);
        LogError(objects[0]->getPose(), ground_truth[frame_first], 0);
      }
      if (mode == MODE_DEMO || mode == MODE_DEMO_AUTO) {
        cv::namedWindow("result", WINDOW_AUTOSIZE);
      }

      int key = (mode == MODE_DEMO_AUTO)? (int)'1' : 0;
      int timeout = 0;
      bool showHelp = true;
      Mat frame;
      const auto start = chrono::steady_clock::now();
      for (int i = frame_first + static_cast<size_t>(mode == MODE_TESTING); i <= frame_last; i++) {
        // obtain an input image
        frame = cv::imread(images[i]);
        if (frame.empty()) {
          std::cerr << "cannot load frame " << images[i] << std::endl;
          continue;
        }
        if (scale != 1.0f) {
          cv::resize(frame, frame, cv::Size(width, height));
        }

        // the main pose update call
        poseEstimator->estimatePoses(frame, false, true);

        // render the models with the resulting pose estimates ontop of the input image
        if (mode == MODE_DEMO || mode == MODE_DEMO_AUTO) {
          Mat result = drawResultOverlay(objects, frame);

          if(showHelp) {
            putText(result, "Press '1' to initialize", Point(150, 250), FONT_HERSHEY_DUPLEX, 1.0, Scalar(255, 255, 255), 1);
            putText(result, "or 'c' to quit", Point(205, 285), FONT_HERSHEY_DUPLEX, 1.0, Scalar(255, 255, 255), 1);
          }

          cv::imshow("result", result);
          if (key != (int)'1') {
            key = cv::waitKey(timeout);
          }

          // start/stop tracking the first object
          if(key == (int)'1') {
            poseEstimator->toggleTracking(frame, 0, false);
            poseEstimator->estimatePoses(frame, false, false);
            timeout = 1;
            showHelp = 0;
            key = 0;
          }
          if(key == (int)'2') {// the same for a second object
            //poseEstimator->toggleTracking(frame, 1, false);
            //poseEstimator->estimatePoses(frame, false, false);
          }
          // reset the system to the initial state
          if(key == (int)'r') {
            poseEstimator->reset();
            timeout = 0;
            showHelp = 1;
            key = (mode == MODE_DEMO_AUTO)? (int)'1' : 0;
            i = frame_first + static_cast<size_t>(mode == MODE_TESTING) - 1;
          }
        }

        for(int count = 1; key != 0x1b && count < iterations; count++) {
          // the main pose update call
          poseEstimator->estimatePoses(frame, false, true);
        }

        if((mode == MODE_DEMO || mode == MODE_DEMO_AUTO) && key == 0x1b) {
            break;
        } else if (mode == MODE_TESTING) {
          LogError(objects[0]->getPose(), ground_truth[i], i);
        }
      }

      const auto end = chrono::steady_clock::now();
      const auto nsecs = static_cast<double>(chrono::duration_cast<chrono::nanoseconds>(end - start).count());
      std::cerr << "\n execution time: " << std::setprecision(3) << (nsecs/1e9) << " seconds"
                << "\n     frame rate: " << std::setprecision(3) << ((frame_last - frame_first + 1) * 1e9 / nsecs) << std::endl;
    }

    // deactivate the offscreen rendering OpenGL context
    RenderingEngine::Instance()->doneCurrent();

    // clean up
    RenderingEngine::Instance()->destroy();

    for(int i = 0; i < objects.size(); i++)
    {
        delete objects[i];
    }
    objects.clear();

    delete poseEstimator;
}
