from django.shortcuts import render,redirect,HttpResponse

from app01 import models


def attendence_list(request):
    nid = request.GET.get('nid')
    attendence_list = models.AttendenceInfo.objects.filter(student_id=nid)
    # for item in attendence_list:
    #     print(item.id, item.student_id.id, item.class_id.id, item.result, item.student_name, item.class_name)
    return render(request,'attendence_list_stu.html',{"attendence_list": attendence_list})