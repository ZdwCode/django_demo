"""django_day3 URL Configuration

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/3.2/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
from django.contrib import admin
from django.urls import path
from app01 import views,teacher,student
urlpatterns = [
    path('admin/', admin.site.urls),
    # 登录界面
    path('login/', views.login),

    # 管理员界面 ----- 部门管理
    path('myadmin/depart/list/', views.depart_list),# 部门展示
    path('myadmin/depart/add/', views.depart_add),# 新增部门
    path('myadmin/depart/edit/', views.depart_edit),# 编辑部门
    path('myadmin/depart/delete/', views.depart_delete),# 删除部门
    # 管理员界面 ----- 用户管理
    path('myadmin/user/list/', views.user_list),# 用户展示
    path('myadmin/user/add/', views.user_add),# 新增用户
    path('myadmin/user/edit/', views.user_edit),# 编辑用户
    path('myadmin/user/delete/', views.user_delete),# 删除用户
    # 管理员界面 ----- 图像信息采集管理
    path('myadmin/pic/', views.ope_pic),# 图像采集
    path('myadmin/face/', views.face_train), # 人脸训练

    #======================= 管理员界面结束 ===============#

    # 教师页面 -----成绩
    path('teacher/attendence/list/', teacher.attendence_list),# 展示出勤记录
    path('teacher/attendence/add/', teacher.attendence_add),# 新建出勤记录
    path('teacher/attendence/edit/', teacher.attendence_edit),# 新建出勤记录
    path('teacher/attendence/delete/', teacher.attendence_delete),# 新建出勤记录
    path('teacher/attendence/start/', teacher.attendence_start),# 新建出勤记录
    path('teacher/course/list/', teacher.course_list),
    path('teacher/course/add/', teacher.course_add),
    path('teacher/course/delete/', teacher.course_delete),
    # 学生页面
    path('student/attendence/list/', student.attendence_list)
]
