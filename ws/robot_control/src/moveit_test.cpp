#include <memory>
#include <rclcpp/rclcpp.hpp>
#include <moveit/move_group_interface/move_group_interface.h>
#include <geometry_msgs/msg/pose.hpp>
#include <tf2/LinearMath/Quaternion.h>
#include <chrono>
#include <cmath>
#include <geometry_msgs/msg/point.hpp>
#include "tf2/LinearMath/Quaternion.h"
#include "tf2_ros/static_transform_broadcaster.h"
#include "geometry_msgs/msg/pose_array.hpp"
#include <visualization_msgs/msg/marker.hpp>
#include <visualization_msgs/msg/marker_array.hpp>
#include <sstream>


using moveit::planning_interface::MoveGroupInterface;
namespace {
    // Only can use std string until cpp20
    constexpr int kDebugArrowId = 10;

    template <class T>
    std::ostream &operator<<(std::ostream &os, const std::vector<T> &v) {
    os << "[";
    for (typename std::vector<T>::const_iterator ii = v.begin(); ii != v.end();
        ++ii) {
        os << " " << *ii;
    }
    os << "]";
    return os;
    }
}
static const std::string PLANNING_GROUP = "panda_arm";

class FrankaMoveItSingle{
public:
    FrankaMoveItSingle(rclcpp::Node::SharedPtr node_ptr)
    : node_ptr(node_ptr),
    logger(node_ptr->get_logger()),
    move_group_interface(MoveGroupInterface(node_ptr, PLANNING_GROUP)){
        RCLCPP_INFO_STREAM(logger, "In the constructor");
    
    move_group_interface.setPlanningTime(10.0);  // Increase time to 10 seconds
    move_group_interface.setPlannerId("RRTConnectkConfigDefault");

    node_ptr->declare_parameter<double>("rate", 0.1);
    rate_hz = node_ptr->get_parameter("rate").as_double();
    std::chrono::milliseconds rate = (std::chrono::milliseconds) ((int)(1000. / rate_hz));
    timer_ = node_ptr->create_wall_timer(
        rate, std::bind(&FrankaMoveItSingle::timer_callback, this)
    );

    }
private:
    // Timer callback function to be called at the "rate" defined
    void timer_callback(){
        RCLCPP_INFO_STREAM(logger, "In the timer callback");
        // Set the target pose
        geometry_msgs::msg::Pose target_pose;
        tf2::Quaternion target_q;
        // target_q.setRPY(0.0, 0.0, 0.0);
        target_pose.position.x = 0.306913;
        target_pose.position.y = -0.354571;
        target_pose.position.z = 0.590295;
        target_pose.orientation.x = 0.92388;
        target_pose.orientation.y = -0.38268;
        target_pose.orientation.z = 1.12945e-5;
        target_pose.orientation.w = 4.36084e-5;
        move_group_interface.setPoseTarget(target_pose, ee_link_name);
        
        moveit::planning_interface::MoveGroupInterface::Plan msg;
        auto const success = static_cast<bool>(move_group_interface.plan(msg));
        auto const plan = msg;
        
        if(success) {
            RCLCPP_INFO(logger, "Planning succeeded, executing...");
            move_group_interface.execute(plan);
        } else {
        RCLCPP_ERROR(logger, "Planing failed!");
        }



    }

    rclcpp::Node::SharedPtr node_ptr;
    rclcpp::TimerBase::SharedPtr timer_;
    double rate_hz;
    rclcpp::Logger logger;
    moveit::planning_interface::MoveGroupInterface move_group_interface;
    std::string ee_link_name;
};

int main(int argc, char* argv[]) {
    rclcpp::init(argc, argv);
    auto const node = std::make_shared<rclcpp::Node>(
        "moveit_test",
        rclcpp::NodeOptions().automatically_declare_parameters_from_overrides(
            true));
    auto const moveit_test = std::make_shared<FrankaMoveItSingle>(node);
    rclcpp::spin(node);
    rclcpp::shutdown();
    return 0;
}