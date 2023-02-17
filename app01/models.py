
from django.db import models

# Create your models here.
class UserInfo(models.Model):
    """ 用户信息表 """
    name = models.CharField(verbose_name="姓名", max_length=32)
    gender_choices = {
        (1, '男'),
        (2, '女')
    }
    gender = models.SmallIntegerField(verbose_name="性别", choices=gender_choices)
    age = models.IntegerField(verbose_name="年龄", default=None)
    depart = models.ForeignKey(verbose_name="所属学院", to="Department", to_field="id", on_delete=models.CASCADE)
    phone = models.IntegerField(verbose_name="电话",default=None)
    type_choices = {
        (1, '管理员'),
        (2, '教师'),
        (3, '学生'),
    }
    type = models.SmallIntegerField(verbose_name="用户类型", choices=type_choices)
    number = models.CharField(verbose_name="账号", max_length=32)
    password = models.CharField(verbose_name="密码", max_length=32)


class CourseInfo(models.Model):
    name = models.CharField(verbose_name="课程名称", max_length=32)
    credit = models.IntegerField(verbose_name="课程学分",default=2)
    start_time = models.DateTimeField(verbose_name="课程开始时间")
    end_time = models.DateTimeField(verbose_name="课程结束时间")
    place = models.ForeignKey(verbose_name="上课地点", to="PlaceInfo", to_field="id", on_delete=models.CASCADE)


class PlaceInfo(models.Model):
    name = models.CharField(verbose_name="地点名称", max_length=32)
    def __str__(self):
        return self.name


class ClassInfo(models.Model):
    start_time = models.DateTimeField(verbose_name="课程开始时间")
    end_time = models.DateTimeField(verbose_name="课程结束时间")
    user = models.CharField(verbose_name="使用者", max_length=32)


class AttendenceInfo(models.Model):
    student_id = models.ForeignKey(verbose_name="学生号", to="UserInfo", to_field="id", on_delete=models.CASCADE)
    class_id = models.ForeignKey(verbose_name="课程号", to="CourseInfo", to_field="id", on_delete=models.CASCADE)
    result = models.IntegerField(verbose_name="是否出勤", default=0)
    student_name = models.CharField(verbose_name="学生姓名",max_length=32)
    class_name = models.CharField(verbose_name="课程名称",max_length=32)


class Department(models.Model):
    """
    部门表
    """
    title = models.CharField(verbose_name="学院名称",max_length=32)
    def __str__(self):
        return self.title
    # password = models.CharField(max_length=32)
    # age = models.IntegerField()