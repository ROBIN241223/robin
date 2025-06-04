<?php
$target_dir = "uploads/";  // Thư mục lưu ảnh (phải tạo thư mục này)
$target_file = $target_dir . basename($_FILES["image"]["name"]);
$uploadOk = 1;
$imageFileType = strtolower(pathinfo($target_file,PATHINFO_EXTENSION));

// Kiểm tra xem có phải là ảnh thật không
if(isset($_POST["submit"])) {
    $check = getimagesize($_FILES["image"]["tmp_name"]);
    if($check !== false) {
        echo "File is an image - " . $check["mime"] . ".";
        $uploadOk = 1;
    } else {
        echo "File is not an image.";
        $uploadOk = 0;
    }
}

// Kiểm tra file đã tồn tại chưa
if (file_exists($target_file)) {
    echo "Xin lỗi, file đã tồn tại.";
    $uploadOk = 0;
}

// Giới hạn kích thước file
if ($_FILES["image"]["size"] > 5000000) { // Ví dụ: 5MB
    echo "Xin lỗi, file quá lớn.";
    $uploadOk = 0;
}

// Chỉ cho phép một số định dạng ảnh
if($imageFileType != "jpg" && $imageFileType != "png" && $imageFileType != "jpeg"
&& $imageFileType != "gif" ) {
    echo "Xin lỗi, chỉ cho phép file JPG, JPEG, PNG & GIF.";
    $uploadOk = 0;
}

 // Kiểm tra nếu $uploadOk = 0, có lỗi
 if ($uploadOk == 0) {
     echo "Xin lỗi, file của bạn không được upload.";
 // Nếu mọi thứ ok, upload file
 } else {
     if (move_uploaded_file($_FILES["image"]["tmp_name"], $target_file)) {
         echo "File ". htmlspecialchars( basename( $_FILES["image"]["name"])). " đã được upload.";

         // Lưu thông tin vào cơ sở dữ liệu (nếu cần)
          // ... (code kết nối và lưu vào database) ...
           // Ví dụ:
           $name = $_POST["name"];
           $description = $_POST["description"];
           // ...

         // Gọi hàm xử lý ảnh (tìm người mất tích)
         //require_once 'find_missing_person.py'; // Nếu dùng Python
         // ... (code để gọi hàm và truyền đường dẫn ảnh $target_file) ...
     } else {
         echo "Xin lỗi, có lỗi khi upload file.";
     }
 }
 ?>