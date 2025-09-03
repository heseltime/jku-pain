"""
URL configuration for bumpserver project.

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/4.2/topics/http/urls/
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
from django.urls import path, re_path
from bump.views import bumpmap, cloudmap, earthtexture, planetary_pain_text

urlpatterns = [
    path('admin/', admin.site.urls),
    path('api/bumpmap/', bumpmap, name='bumpmap'),     # with slash
    re_path(r'^api/bumpmap$', bumpmap),                # without slash
    path('api/cloudmap/', cloudmap),
    re_path(r'^api/cloudmap$', cloudmap),
    path('api/earthtexture/', earthtexture),
    re_path(r'^api/earthtexture$', earthtexture),
    path('api/planetary-pain/', planetary_pain_text, name='planetary_pain_text'),
    re_path(r'^api/planetary-pain$', planetary_pain_text),
]
