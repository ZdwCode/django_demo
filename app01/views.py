import os

from django.shortcuts import render,redirect,HttpResponse

# Create your views here.
# 登录界面
from app01 import models
def login(request):
    if request.method == "GET":
        return render(request,'login.html')
    user = request.POST.get('user')
    password = request.POST.get('password')
    item = models.UserInfo.objects.filter(number=user).first()
    if item is None:
        return HttpResponse('用户不存在错误')
    if item.password == password:
        type = item.type
        if type == 1:
            return redirect('/myadmin/depart/list/')
        elif type == 2:
            return redirect('/teacher/attendence/list/')
        elif type == 3:
            # return redirect(f'/student/attendence/list/?nid={item.id}')
            return redirect(f'/student/attendence/list/?nid={item.id}')
    return HttpResponse('账号或密码错误')
# 管理员的界面
def depart_list(request):
    depart_list = models.Department.objects.all()
    return render(request, 'department_list.html', {"depart_list": depart_list})

def depart_add(request):
    """添加部门"""
    #return HttpResponse("添加部门")
    if request.method == "GET":
        return render(request,'department_add.html')

    # 获取用户提交的数据
    name = request.POST.get("name")
    models.Department.objects.create(title=name)
    return redirect('/myadmin/depart/list/')

def depart_edit(request):
    nid = request.GET.get('nid')
    if request.method == "GET":
        row_object = models.Department.objects.filter(id=nid).first()
        title = row_object.title
        return render(request,'department_edit.html',{"title":title})
    title = request.POST.get("name")
    models.Department.objects.filter(id=nid).update(title=title)

    return redirect("/myadmin/depart/list/")

def depart_delete(request):
    nid = request.GET.get('nid')
    models.Department.objects.filter(id=nid).delete()
    return redirect('/myadmin/depart/list/')

def user_list(request):
    user_list = models.UserInfo.objects.all()
    # for item in user_list:
    #     print(item.id,item.name,item.account,item.create_time.strftime('%Y-%m-%d'),item.get_gender_display(),item.depart.title)
    return render(request, 'user_list.html', {"user_list": user_list})


from django import forms
class UserModelForm(forms.ModelForm):
    class Meta:
        model = models.UserInfo
        fields = ['name', 'gender', 'age','depart','phone','type','number','password']
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        # 找到所有插件
        for name,filed in self.fields.items():
            filed.widget.attrs = {"class": "form-control"}
            if name == "password":
                filed.widget.attrs = {"class": "form-control"}


def user_add(request):
    if request.method == "GET":
        form = UserModelForm()  # instance参数
        return render(request, 'user_add.html',{'form':form})
    form = UserModelForm(data=request.POST)
    if form.is_valid():
        form.save()
        return redirect('/myadmin/user/list/')
    return HttpResponse('失败新增',form.errors)

def user_edit(request):
    if request.method == "GET":
        nid = request.GET.get('nid')
        instance = models.UserInfo.objects.filter(id=nid).first()
        form = UserModelForm(instance=instance)  # instance参数
        return render(request, 'user_edit.html', {'form': form})
    nid = request.GET.get('nid')
    instance = models.UserInfo.objects.filter(id=nid).first()
    form = UserModelForm(data=request.POST,instance=instance)
    if form.is_valid():
        form.save()
        return redirect('/myadmin/user/list/')
    return HttpResponse('失败新增', form.errors)

def user_delete(request):
    nid = request.GET.get('nid')
    models.UserInfo.objects.filter(id=nid).delete()
    return redirect('/myadmin/user/list/')

from app01.face_recognize import get_face
def ope_pic(request):
    # get_face.get_face()
    if request.method == "GET":
        return render(request,'pic.html')
    name = request.POST.get('name')
    get_face.get_face(name)
    return redirect('/myadmin/user/list/')
from app01.face_recognize.TransferModel import case
def face_train(request):
    mytype = request.GET.get('type')
    if mytype == '1':
        meta = []
        for i in range(len(os.listdir('./app01/face_recognize/face_image/data/'))):
            item = {
                "name":os.listdir('./app01/face_recognize/face_image/data/')[i]
            }
            meta.append(item)
        return render(request,'face_reco.html',{'meta':meta})
    case.start()
    return redirect('/myadmin/user/list/')
