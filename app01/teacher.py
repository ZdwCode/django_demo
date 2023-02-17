from django.shortcuts import render,redirect,HttpResponse

# Create your views here.
# 登录界面
from app01 import models

def attendence_list(request):
    if request.method == "GET":
        attendence_list = models.AttendenceInfo.objects.all()
        # for item in attendence_list:
        #     print(item.id,item.student_id.id,item.class_id.id,item.result,item.student_name,item.class_name)
        return render(request,'attendence_list.html', {"attendence_list": attendence_list})
    student_name = request.POST.get('student_name')
    course_name = request.POST.get('course_name')
    attendence_list = models.AttendenceInfo.objects.filter(student_name__contains=student_name,class_name__contains=course_name)
    return render(request, 'attendence_list.html', {"attendence_list": attendence_list})
from django import forms
class UserModelForm(forms.ModelForm):
    class Meta:
        model = models.AttendenceInfo
        fields = ['student_id', 'class_id', 'result','student_name','class_name']
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        # 找到所有插件
        for name,filed in self.fields.items():
            filed.widget.attrs = {"class": "form-control"}
def attendence_add(request):
    if request.method == "GET":
        return render(request, 'attendence_add.html')
    student_id = request.POST.get("student_id")
    class_id = request.POST.get("class_id")
    student_name = request.POST.get("student_name")
    course_name = request.POST.get("course_name")
    result = request.POST.get("result")
    student_id = models.UserInfo.objects.filter(id=student_id).first()
    class_id = models.CourseInfo.objects.filter(id=class_id).first()
    models.AttendenceInfo.objects.create(student_id=student_id,
                                         class_id=class_id,
                                         student_name=student_name,
                                         class_name=course_name,
                                         result=result)
    return redirect('/teacher/attendence/list/')

def attendence_edit(request):
    nid = request.GET.get('nid')
    if request.method == "GET":
        row_object = models.AttendenceInfo.objects.filter(id=nid).first()
        meta = {
            "id":row_object.id,
            "result":row_object.result,
            "student_name":row_object.student_name,
            "class_name":row_object.class_name,
            "class_id":row_object.class_id.id,
            "student_id":row_object.student_id.id,
        }
        return render(request, 'attendence_edit.html', {"meta": meta})
    result = request.POST.get("result")
    student_name = request.POST.get("student_name")
    class_name = request.POST.get("course_name")
    class_id = request.POST.get("class_id")
    student_id = request.POST.get("student_id")
    student_id = models.UserInfo.objects.filter(id=student_id).first()
    class_id = models.CourseInfo.objects.filter(id=class_id).first()
    models.AttendenceInfo.objects.filter(id=nid).update(result=result,
                                                        student_name=student_name,
                                                        student_id=student_id,
                                                        class_id=class_id,
                                                        class_name=class_name)
    return redirect('/teacher/attendence/list/')

def attendence_delete(request):
    nid = request.GET.get('nid')
    models.AttendenceInfo.objects.filter(id=nid).delete()
    return redirect('/teacher/attendence/list/')
from app01.face_recognize import recoginize
def attendence_start(request):
    if request.method == 'GET':
        course_list = models.CourseInfo.objects.all()
        for item in course_list:
            print(item.id, item.name)
        return render(request,'attendence_start.html', {"course_list":course_list})
    class_id = request.POST.get("class_id")
    class_name = request.POST.get("class_name")
    class_id = models.CourseInfo.objects.filter(id=class_id).first()
    id_list = recoginize.start()
    print(id_list,'here')
    for id in id_list:
        row_object = models.UserInfo.objects.filter(id=id).first()
        if row_object:
            student_name = row_object.name
            #print(id,student_name,class_name,class_id)
            student_id = models.UserInfo.objects.filter(id=id).first()
            models.AttendenceInfo.objects.filter(student_id=student_id,class_id=class_id).update(
                                            result='1')
    return redirect('/teacher/attendence/list/')

def course_list(request):
    if request.method == "GET":
        course_list = models.CourseInfo.objects.all()
        return render(request, 'course_list.html', {"course_list": course_list})
    course_name = request.POST.get('course_name')
    course_list = models.CourseInfo.objects.filter(name__contains=course_name)
    return render(request, 'course_list.html', {"course_list": course_list})

from django import forms
class CourseModelForm(forms.ModelForm):
    class Meta:
        model = models.CourseInfo
        fields = ['name', 'credit', 'start_time','end_time','place']
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        # 找到所有插件
        for name,filed in self.fields.items():
            filed.widget.attrs = {"class": "form-control"}


def course_add(request):
    if request.method == "GET":
        form = CourseModelForm()  # instance参数
        return render(request, 'course_add.html', {'form': form})
    form = CourseModelForm(data=request.POST)  # instance参数
    if form.is_valid():
        form.save()
        return redirect('/teacher/course/list/')
    return HttpResponse('失败新增',form.errors)

def course_delete(request):
    nid = request.GET.get('nid')
    models.CourseInfo.objects.filter(id=nid).delete()
    return redirect('/teacher/course/list/')