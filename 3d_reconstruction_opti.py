# --------------------------------------------
# Structure-from-Motion Pipeline (PA1 full code)
# Annotated version following the PDF instructions
# Steps: I to VII (Feature Matching -> Camera Estimation -> Triangulation -> Multi-view SfM)
# --------------------------------------------

# Step 0: Load libraries and dependencies (as permitted by the assignment)
# Note: OpenCV is the main library for computer vision, Open3D is used only for visualization and saving .ply files

import open3d as o3d
import numpy as np
import cv2
import os
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# ----------------------------------------------------------------
# Step I: Feature Extraction & Matching using SIFT and BFMatcher
# ----------------------------------------------------------------
def sift_features(img_1_color, img_2_color, cv2_COLOR2GRAY):
    img_1_gray = cv2.cvtColor(img_1_color, cv2_COLOR2GRAY)
    img_2_gray = cv2.cvtColor(img_2_color, cv2_COLOR2GRAY)

    sift = cv2.SIFT_create()
    key_pts_0, desc_0 = sift.detectAndCompute(img_1_gray, None)
    key_pts_1, desc_1 = sift.detectAndCompute(img_2_gray, None)

    return key_pts_0, desc_0, key_pts_1, desc_1

# ----------------------------------------------------------------
# Detects and computes keypoints + descriptors using SIFT
# ----------------------------------------------------------------
def common_correspondences(key_pts_0, desc_0, key_pts_1, desc_1):
        bf = cv2.BFMatcher()
        matches = bf.knnMatch(desc_0, desc_1, k=2)
        feature = []
        for m, n in matches:
            # Ratio test Lowe is useful to remove outlier (GPT)
            if m.distance < 0.70 * n.distance:
                feature.append(m)

        # We gather the commons features from the two images
        feature_0 = [key_pts_0[m.queryIdx].pt for m in feature]
        feature_1 = [key_pts_1[m.trainIdx].pt for m in feature]

        return np.float32(feature_0), np.float32(feature_1)

# -----------------------------------------------------
# Step II: Estimate Essential Matrix with RANSAC filtering
# -----------------------------------------------------
def manual_recover_pose(E, pts1, pts2, K):
    # Decomposes E -> 4 poses -> selects best one (Step III)

    # Normalize points
    pts1_norm = cv2.undistortPoints(np.expand_dims(pts1, axis=1), K, None)
    pts2_norm = cv2.undistortPoints(np.expand_dims(pts2, axis=1), K, None)

    R1, R2, t = cv2.decomposeEssentialMat(E)
    poses = [(R1, t), (R1, -t), (R2, t), (R2, -t)]

    P0 = np.hstack((np.eye(3), np.zeros((3, 1))))
    best_count = -1
    best_pose = None
    best_mask = None

    for R, t_vec in poses:
        P1 = np.hstack((R, t_vec))
        # Triangulation
        pts_4d = cv2.triangulatePoints(P0, P1, pts1_norm.squeeze().T, pts2_norm.squeeze().T)
        pts_3d = pts_4d[:3] / pts_4d[3]

        # Points in front of the camera
        z1 = pts_3d[2]
        z2 = (R[2] @ pts_3d + t_vec[2]).reshape(-1)
        mask = (z1 > 0) & (z2 > 0)
        count = np.sum(mask)

        if count > best_count:
            best_count = count
            best_pose = (R, t_vec)
            best_mask = mask.astype(np.uint8).reshape(-1, 1)

    return best_pose[0], best_pose[1], best_mask

# ---------------------------------------------------------------------
# Step III: Reprojection test to verify camera pose & point consistency
# ---------------------------------------------------------------------
def compute_reprojection_rmse(pts_3d, pts_2d, pose_matrix, intrinsics):
    # Projects 3D points and compares them to image observations
    R = pose_matrix[:, :3]
    t = pose_matrix[:, 3]

    pts_3d = cv2.convertPointsFromHomogeneous(pts_3d.T)
    rvec, _ = cv2.Rodrigues(R)
    tvec = t.reshape(3, 1)

    # Projection of 3D points to 2D
    proj_pts_2d, _ = cv2.projectPoints(pts_3d, rvec, tvec, intrinsics, None)
    proj_pts_2d = np.float32(proj_pts_2d[:, 0, :])

    # Error calculation
    diff = proj_pts_2d - pts_2d
    squared_error = np.sum(diff**2, axis=1)

    # RMSE (Root Mean Square Error)
    rmse = np.sqrt(np.mean(squared_error))

    return rmse / len(proj_pts_2d), pts_3d

# ------------------------------------------
# Step IV: Triangulation of 3D points (2-view)
# Is done inside the code with cv2.triangulatePoints
# ------------------------------------------

# --------------------------------------------------------------
# Step V: Growing step — Multi-view pose estimation with PnP
# --------------------------------------------------------------
def PnP(pts_3d, pts_2d , K, dist_coeff, r, is_initialization):
        # Perspective-n-Point algorithm to estimate pose from 2D-3D
        if is_initialization:
            pts_3d = pts_3d[:, 0 ,:]
        _, rot_vector_calc, t, inlier = cv2.solvePnPRansac(pts_3d, pts_2d, K, dist_coeff, cv2.SOLVEPNP_ITERATIVE)
        # Converts a rotation matrix to a rotation vector or vice versa
        R, _ = cv2.Rodrigues(rot_vector_calc)

        if inlier is not None:
            pts_2d = pts_2d[inlier[:, 0]]
            pts_3d = pts_3d[inlier[:, 0]]
            r = r[inlier[:, 0]]
        return R, t, pts_2d, pts_3d, r

# Helper: Find overlapping points between 3 images (for chaining)
def common_points(pts1, pts2, pts3):
    common_1, common_2 = [], []

    for i in range(pts1.shape[0]):
        matches = np.where((pts2 == pts1[i]).all(axis=1))[0]
        if matches.size > 0:
            common_1.append(i)
            common_2.append(matches[0])

    mask_pts2 = np.delete(pts2, common_2, axis=0)
    mask_pts3 = np.delete(pts3, common_2, axis=0)

    print("Shape New Arrays:", mask_pts2.shape, mask_pts3.shape)
    return np.array(common_1), np.array(common_2), mask_pts2, mask_pts3

# -------------------------------------------------------------
# Step VI: Reprojection error and filtering of good observations
# -------------------------------------------------------------
def compute_reprojection_rmse(pts_3d, pts_2d, pose_matrix, K):
    R = pose_matrix[:, :3]
    t = pose_matrix[:, 3]
    
    pts_3d = cv2.convertPointsFromHomogeneous(pts_3d.T)

    r, _ = cv2.Rodrigues(R)
    t = t.reshape(3, 1)

    # Projection of 3D points to 2D
    proj_pts_2d, _ = cv2.projectPoints(pts_3d, r, t, K, None)
    proj_pts_2d = np.float32(proj_pts_2d[:, 0, :])

    # Error calculation
    diff = proj_pts_2d - pts_2d
    squared_error = np.sum(diff**2, axis=1)

    # RMSE (Root Mean Square Error)
    rmse = np.sqrt(np.mean(squared_error))

    return rmse / len(proj_pts_2d), pts_3d

def save_points_if_valid(pts_3d, pts_2d, img, total_pts, total_colors, error, threshold=10):
    """
    Ajoute les points 3D et leurs couleurs si l'erreur de reprojection est inférieure au seuil.
    """
    if error < threshold:
        # Pixel position must be an integer (and SIFT gives us float)
        pts_2d = np.round(pts_2d).astype(int)
        # Ensure that the rounded coordinate doesn't excede the image's dimension
        pts_2d = np.clip(pts_2d, [0, 0], [img.shape[1] - 1, img.shape[0] - 1])
        # Recording of colors
        colors = np.array([img[pt[1], pt[0]] for pt in pts_2d])
        total_colors = np.vstack((total_colors, colors))
        # Recording of 3d points
        total_pts = np.vstack((total_pts, pts_3d[:, 0, :]))
    else:
        print("------------------\nError to high, points not recorded\n------------------")

    return total_pts, total_colors

# -------------------------------------------
# Step VII: Save intermediate reconstruction
# -------------------------------------------
def save_step_ply(step_num, pts_3d, colors, folder):
    os.makedirs(folder, exist_ok=True)
   
    # Filter the main cluster (otherwise, result is impossible to visualize)
    pts_3d, colors = filter_main_cluster(pts_3d, colors)
    filtered_pts = o3d.geometry.PointCloud()
    filtered_pts.points = o3d.utility.Vector3dVector(pts_3d)
    filtered_pts.colors = o3d.utility.Vector3dVector(colors / 255.0)

    ply_path = os.path.join(folder, f"points_step_{step_num:02d}.ply")
    o3d.io.write_point_cloud(ply_path, filtered_pts)


# ------------------------------------------------------------------
# Step VII (Final Output): Plot 3D points and estimated camera poses
# ------------------------------------------------------------------
def plot_3d_with_cameras(pts_3d, poses):
    poses = np.array(poses)
    pts = np.asarray(pts_3d.points)

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.set_title("3D Points & Camera Trajectory")

    # Points cloud
    ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], s=0.5, c='gray', alpha=0.5)

    # Cameras
    ax.scatter(poses[:, 0], poses[:, 1], poses[:, 2],
               c='red', label='Cameras')
    ax.plot(poses[:, 0], poses[:, 1], poses[:, 2],
            c='red', linewidth=1)

    ax.legend()
    plt.tight_layout()
    plt.show()

# -----------------------------------------------------
# Utility: Keep only the dominant cluster (filter outliers)
# -----------------------------------------------------
def filter_main_cluster(pts_3d, colors, eps=0.3, min_samples=10):
    clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(pts_3d)
    labels = clustering.labels_

    # Cluster principal = label le plus fréquent (hors bruit -1)
    unique, counts = np.unique(labels[labels != -1], return_counts=True)
    if len(unique) == 0:
        print("No valid cluster found.")
        return pts_3d, colors

    main_label = unique[np.argmax(counts)]
    mask = labels == main_label

    return pts_3d[mask], colors[mask]



# ------------------
# MAIN PROGRAM
# ------------------

print("\nWhich dataset do you need")
print("1 : Dataset from the PA")
print("2 : Dataset from the student")
nb = input("Type '1', or '2' : ")

# Variables to access the dataset
match nb:
    case '1':
        data_path = "./data_prof"
        ply_dir = "./ply_prof"
    case '2':
        data_path = "./data_personal"
        ply_dir = "./ply_personal"
    case _:
        print("You have entered an incorect value")

# Loading of images
img_names = sorted([f for f in os.listdir(data_path) if f.endswith(('.JPG', '.jpg'))])

# Re-ordering the profesor dataset
if nb == '1':
    temp = img_names[::-1]
    temp[16:] = img_names[:16]
    img_names = temp
    cv2_COLOR2GRAY = cv2.COLOR_BGR2GRAY
else:
    cv2_COLOR2GRAY = cv2.COLOR_RGB2GRAY

print(f"{len(img_names)} images loaded")

# Loading of K
K = np.loadtxt(os.path.join(data_path, "K.txt"), dtype=np.float32)
print(f"\n{K = }\n")

# Definition of transformation matrix for keeping tracks of the reconstructed 3D and their colors
M_0 = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]])
M_1 = np.empty((3, 4))
total_pts = np.zeros((1,3))
total_colors = np.zeros((1,3))
camera_positions = [np.zeros(3)]# Camera at origine

# Load the first two images
img_0 = cv2.imread(os.path.join(data_path, img_names[0]), cv2.IMREAD_COLOR)
img_1 = cv2.imread(os.path.join(data_path, img_names[1]), cv2.IMREAD_COLOR)
# Color conversion for my dataset
if nb == '2':
        img_0 = cv2.cvtColor(img_0, cv2.COLOR_BGR2RGB)
        img_1 = cv2.cvtColor(img_1, cv2.COLOR_BGR2RGB)

# Step I:  SIFT + matching + essential matrix with RANSAC

# Find the SIFT features of the 2 first images
key_pts_0, desc_0, key_pts_1, desc_1 = sift_features(img_0, img_1, cv2_COLOR2GRAY)

# Find the correspondances for the SIFT feaetures
feat_0, feat_1 = common_correspondences(key_pts_0, desc_0, key_pts_1, desc_1)

# Essential matrix
E, e_mask = cv2.findEssentialMat(feat_0, feat_1, K, method=cv2.RANSAC, prob=0.999, threshold=0.4, mask=None)
feat_0 = feat_0[e_mask.ravel() == 1]
feat_1 = feat_1[e_mask.ravel() == 1]

# Step II and III : Decompose E -> (R, t) and verify the combo
R, t, e_mask = manual_recover_pose(E, feat_0, feat_1, K)
feat_0 = feat_0[e_mask.ravel() > 0]
feat_1 = feat_1[e_mask.ravel() > 0]
M_1[:3, :3] = np.matmul(R, M_0[:3, :3])
M_1[:3, 3] = M_0[:3, 3] + np.matmul(M_0[:3, :3], t.ravel())

pose_0 = np.matmul(K, M_0)
pose_1 = np.matmul(K, M_1)

# Step IV: Triangulate initial 3D points
pts_3d = cv2.triangulatePoints(pose_0, pose_1, feat_0.T, feat_1.T)
pts_3d /= pts_3d[3]

error, pts_3d = compute_reprojection_rmse(pts_3d, feat_1, M_1, K)
#ideally error < 1
print("REPROJECTION ERROR: ", error)

# Step V: Run PnP to refine pose
_, _, feat_1, pts_3d, _ = PnP(pts_3d, feat_1, K, np.zeros((5, 1), dtype=np.float32), feat_0, is_initialization=True)

# Step VI: Reprojection filtering & record 3D points/colors
pts_2d = np.round(feat_1).astype(int)
# Ensure that the rounded coordinate doesn't excede the image's dimension
pts_2d = np.clip(pts_2d, [0, 0], [img_1.shape[1] - 1, img_1.shape[0] - 1])
# Recording of colors
colors = np.array([img_1[pt[1], pt[0]] for pt in pts_2d])
total_colors = np.vstack((total_colors, colors))
total_pts = np.vstack((total_pts, pts_3d))

# Step VII: Save initial .ply file
save_step_ply(1, total_pts, total_colors, ply_dir)

total_img = len(img_names) - 2 

threshold = 0.5
for i in range(total_img):
    print(f"Image {i+2}/{len(img_names)-1}")
    img_2 = cv2.imread(os.path.join(data_path, img_names[i + 2]))
    if nb == '2':
        img_2 = cv2.cvtColor(img_2, cv2.COLOR_BGR2RGB)

    key_pts_1, desc_1, key_pts_2, desc_2 = sift_features(img_1, img_2, cv2_COLOR2GRAY)
    feat_cur, feat_2 = common_correspondences(key_pts_1, desc_1, key_pts_2, desc_2)

    if i != 0:
        # Step IV: Triangulate initial 3D points
        pts_3d = cv2.triangulatePoints(pose_0, pose_1, feat_0.T, feat_1.T)
        pts_3d /= pts_3d[3]
        pts_3d = cv2.convertPointsFromHomogeneous(pts_3d.T)
        pts_3d = pts_3d[:, 0, :]


    cm_pts_0, cm_pts_1, cm_mask_0, cm_mask_1 = common_points(feat_1, feat_cur, feat_2)
    cm_pts_2 = feat_2[cm_pts_1]
    cm_pts_cur = feat_cur[cm_pts_1]

    # Step V: Run PnP to refine pose
    R, t, cm_pts_2, pts_3d, cm_pts_cur = PnP(pts_3d[cm_pts_0], cm_pts_2, K, np.zeros((5, 1), dtype=np.float32), cm_pts_cur, is_initialization=False)
    M_1 = np.hstack((R, t))
    # Position de la caméra courante (dans le repère monde)
    camera_position = -M_1[:, :3].T @ M_1[:, 3]
    camera_positions.append(camera_position)
    
    pose_2 = np.matmul(K, M_1)

    pts_3d = cv2.triangulatePoints(pose_1, pose_2, cm_mask_0.T, cm_mask_1.T)
    pts_3d /= pts_3d[3]
    error, pts_3d = compute_reprojection_rmse(pts_3d, cm_mask_1, M_1, K)
    print("Reprojection Error: ", error)

    # Recording of colors and points
    total_pts, total_colors = save_points_if_valid(pts_3d, cm_mask_1, img_2, total_pts, total_colors, error)
    save_step_ply(i + 2, total_pts, total_colors, ply_dir)

    M_0 = np.copy(M_1)
    pose_0 = np.copy(pose_1)

    img_0 = np.copy(img_1)
    img_1 = np.copy(img_2)

    feat_0 = np.copy(feat_cur)
    feat_1 = np.copy(feat_2)
    pose_1 = np.copy(pose_2)


# Create an Open3D point cloud object
point_cloud = o3d.geometry.PointCloud()

# Set the points and colors for the point cloud
point_cloud.points = o3d.utility.Vector3dVector(total_pts)
point_cloud.colors = o3d.utility.Vector3dVector(total_colors / 255.0)

# Step VI: Reprojection filtering & record 3D points/colors
total_pts, total_colors = filter_main_cluster(total_pts, total_colors)
filtered_point_cloud = o3d.geometry.PointCloud()
filtered_point_cloud.points = o3d.utility.Vector3dVector(total_pts)
filtered_point_cloud.colors = o3d.utility.Vector3dVector(total_colors / 255.0)

# Save the point cloud to a PLY file and then, visualize it
if nb == '1':
    path_ply = "./res/point_cloud_dataset_1.ply"
else:
    path_ply = "./res/point_cloud_dataset_2.ply"

o3d.io.write_point_cloud(path_ply, filtered_point_cloud)

# Visualization
o3d.visualization.draw_geometries([filtered_point_cloud])

# Step VII: Save initial .ply file
plot_3d_with_cameras(filtered_point_cloud, camera_positions)