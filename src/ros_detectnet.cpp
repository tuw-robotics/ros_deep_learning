#include "ros_detectnet.h"

namespace ros_deep_learning
{
ros_detectnet::~ros_detectnet()
{
  ROS_INFO("\nshutting down...\n");
  if (gpu_data_)
    CUDA(cudaFree(gpu_data_));
  delete net_;
}
void ros_detectnet::onInit()
{
  // get a private nodehandle
  ros::NodeHandle& private_nh = getPrivateNodeHandle();
  // get parameters from server, checking for errors as it goes

  std::string prototxt_path, model_path, mean_binary_path, class_labels_path;
  if (!private_nh.getParam("prototxt_path", prototxt_path))
    ROS_ERROR("unable to read prototxt_path for detectnet node");
  if (!private_nh.getParam("model_path", model_path))
    ROS_ERROR("unable to read model_path for detectnet node");

  // make sure files exist (and we can read them)
  if (access(prototxt_path.c_str(), R_OK))
    ROS_ERROR("unable to read file \"%s\", check filename and permissions", prototxt_path.c_str());
  if (access(model_path.c_str(), R_OK))
    ROS_ERROR("unable to read file \"%s\", check filename and permissions", model_path.c_str());

  // create imageNet
  // net = imageNet::Create(prototxt_path.c_str(),model_path.c_str(),NULL,class_labels_path.c_str());

  // create detectNet
  // net = detectNet::Create(detectNet::PEDNET, 0.5f, 2);
  net_ = detectNet::Create(prototxt_path.c_str(), model_path.c_str(), 0.0f, 0.5f, DETECTNET_DEFAULT_INPUT,
                           DETECTNET_DEFAULT_COVERAGE, DETECTNET_DEFAULT_BBOX, 2);

  if (!net_)
  {
    ROS_INFO("ros_detectnet: failed to initialize detectNet\n");
    return;
  }

  // setup image transport
  image_transport::ImageTransport it(private_nh);
  // subscriber for passing in images
  camsub_ = it.subscribeCamera("imin", 10, &ros_detectnet::cameraCallback, this);

  impub_ = it.advertise("image_out", 1);

  personpub_ = private_nh.advertise<tuw_object_msgs::ObjectDetection>("detected_persons_tuw", 1);

  // publisher for classifier output
  // class_pub = private_nh.advertise<std_msgs::Int32>("class",10);
  // publisher for human-readable classifier output
  // class_str_pub = private_nh.advertise<std_msgs::String>("class_str",10);

  // init gpu memory
  gpu_data_ = NULL;
}

void ros_detectnet::cameraCallback(const sensor_msgs::ImageConstPtr& input,
                                   const sensor_msgs::CameraInfoConstPtr& camera_info)
{
  // camera matrix
  Eigen::Matrix<double, 3, 3, Eigen::RowMajor> K(camera_info->K.data());
  Eigen::Matrix<double, 3, 3, Eigen::RowMajor> K_inv = K.inverse();

  cv::Mat cv_im = cv_bridge::toCvCopy(input, "bgr8")->image;
  cv::Mat cv_result;

  ROS_DEBUG("ros_detectnet: image ptr at %p", cv_im.data);
  // convert bit depth
  cv_im.convertTo(cv_im, CV_32FC3);
  // convert color
  cv::cvtColor(cv_im, cv_im, CV_BGR2RGBA);

  // allocate GPU data if necessary
  if (gpu_data_ == NULL)
  {
    ROS_DEBUG("ros_detectnet: first allocation");
    CUDA(cudaMalloc(&gpu_data_, cv_im.rows * cv_im.cols * sizeof(float4)));
  }
  else if (imgHeight_ != cv_im.rows || imgWidth_ != cv_im.cols)
  {
    ROS_DEBUG("ros_detectnet: re allocation");
    // reallocate for a new image size if necessary
    CUDA(cudaFree(gpu_data_));
    CUDA(cudaMalloc(&gpu_data_, cv_im.rows * cv_im.cols * sizeof(float4)));
  }

  // allocate memory for output bounding boxes
  const uint32_t maxBoxes = net_->GetMaxBoundingBoxes();
  ROS_DEBUG("ros_detectnet: maximum bounding boxes: %u\n", maxBoxes);
  const uint32_t classes = net_->GetNumClasses();

  float* bbCPU = NULL;
  float* bbCUDA = NULL;
  float* confCPU = NULL;
  float* confCUDA = NULL;

  if (!cudaAllocMapped((void**)&bbCPU, (void**)&bbCUDA, maxBoxes * sizeof(float4)) ||
      !cudaAllocMapped((void**)&confCPU, (void**)&confCUDA, maxBoxes * classes * sizeof(float)))
  {
    ROS_ERROR("ros_detectnet: failed to alloc output memory\n");
  }

  int numBoundingBoxes = maxBoxes;

  imgHeight_ = cv_im.rows;
  imgWidth_ = cv_im.cols;
  imgSize_ = cv_im.rows * cv_im.cols * sizeof(float4);
  float4* cpu_data = (float4*)(cv_im.data);

  std::vector<Eigen::Vector3f> center_points;

  // copy to device
  CUDA(cudaMemcpy(gpu_data_, cpu_data, imgSize_, cudaMemcpyHostToDevice));

  bool det_result = net_->Detect((float*)gpu_data_, imgWidth_, imgHeight_, bbCPU, &numBoundingBoxes, confCPU);

  if (det_result)
  {
    int lastClass = 0;
    int lastStart = 0;

    for (int n = 0; n < numBoundingBoxes; n++)
    {
      const int nc = confCPU[n * 2 + 1];
      float* bb = bbCPU + (n * 4);

      ROS_INFO("ros_detectnet: bounding box %i   (%f, %f)  (%f, %f)  w=%f  h=%f\n", n, bb[0], bb[1], bb[2], bb[3],
               bb[2] - bb[0], bb[3] - bb[1]);

      if (nc != lastClass || n == (numBoundingBoxes - 1))
      {
        if (!net_->DrawBoxes((float*)gpu_data_, (float*)gpu_data_, imgWidth_, imgHeight_, bbCUDA + (lastStart * 4),
                             (n - lastStart) + 1, lastClass))
          ROS_ERROR("ros_detectnet: failed to draw boxes\n");

        // calculate ground center of bounding box
        Eigen::Vector3f P1_img;
        P1_img(0) = bb[0] + (bb[2] - bb[0]) / 2;
        P1_img(1) = bb[1];
        P1_img(2) = 1;
        center_points.emplace_back(P1_img);

        // CUDA(cudaNormalizeRGBA((float4*)gpu_data_, make_float2(0.0f, 255.0f), (float4*)gpu_data_, make_float2(0.0f,
        // 1.0f), imgWidth_, imgHeight_));

        // copy back to host
        CUDA(cudaMemcpy(cpu_data, gpu_data_, imgSize_, cudaMemcpyDeviceToHost));

        lastClass = nc;
        lastStart = n;

        CUDA(cudaDeviceSynchronize());
      }
    }
  }
  else
  {
    ROS_ERROR("detectnet: detection error occured");
  }

  Eigen::Vector3f P0(0, 0, 0);
  Eigen::Vector3f P1;
  Eigen::Vector3f P_diff;
  Eigen::Vector3f P3D;

  cv_result = cv::Mat(imgHeight_, imgWidth_, CV_32FC4, cpu_data);
  cv_result.convertTo(cv_result, CV_8UC4);

  cv::cvtColor(cv_result, cv_result, CV_RGBA2BGR);

  tuw_object_msgs::ObjectDetection detected_persons_tuw;
  detected_persons_tuw.header = input->header;

  for (size_t i = 0; i < center_points.size(); i++)
  {
    cv::circle(cv_result, cv::Point(center_points[i](0), center_points[i](1)), 2, cv::Scalar(0, 0, 255), -1, 8, 0);

    // calculate 3D position through intersection with ground plane
    P1 = center_points[i];
    P_diff = P1 - P0;
    float nom = gpd_ - gpn_.dot(P0);
    float denom = gpn_.dot(P_diff);
    P3D = P0 + nom / denom * P_diff;

    tuw_object_msgs::ObjectWithCovariance obj;
    obj.covariance_pose.emplace_back(0.5);
    obj.covariance_pose.emplace_back(0);
    obj.covariance_pose.emplace_back(0);
    obj.covariance_pose.emplace_back(0);
    obj.covariance_pose.emplace_back(0.5);
    obj.covariance_pose.emplace_back(0);
    obj.covariance_pose.emplace_back(0);
    obj.covariance_pose.emplace_back(0);
    obj.covariance_pose.emplace_back(0);

    obj.object.ids.emplace_back(i);
    obj.object.ids_confidence.emplace_back(1.0);
    obj.object.pose.position.x = -P3D(0);
    obj.object.pose.position.y = -P3D(1);
    obj.object.pose.position.z = -P3D(2);
    obj.object.pose.orientation.x = 0.0;
    obj.object.pose.orientation.y = 0.0;
    obj.object.pose.orientation.z = 0.0;
    obj.object.pose.orientation.w = 1.0;

    detected_persons_tuw.objects.emplace_back(obj);
  }

  personpub_.publish(detected_persons_tuw);
  impub_.publish(cv_bridge::CvImage(std_msgs::Header(), "bgr8", cv_result).toImageMsg());
}

void ros_detectnet::groundPlaneCallback(const rwth_perception_people_msgs::GroundPlane::ConstPtr& gp)
{
  gp_ = gp;

  // ground plane normal vector
  gpn_(0) = gp_->n[0];
  gpn_(1) = gp_->n[1];
  gpn_(2) = gp_->n[2];

  // ground plane distance ax+by+cz = d
  double gpd_ = ((double)gp_->d) * (-1000.0);
}

}  // namespace ros_deep_learning
