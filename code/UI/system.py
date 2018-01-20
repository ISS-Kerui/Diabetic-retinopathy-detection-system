# -*- coding: utf-8 -*- 

###########################################################################
## Python code generated with wxFormBuilder (version Aug 23 2015)
## http://www.wxformbuilder.org/
##
## PLEASE DO "NOT" EDIT THIS FILE!
###########################################################################

import wx
import wx.xrc
#import train
import secondframe
import thirdframe
import os

###########################################################################
## Class MyFrame1
###########################################################################

class MyFrame1 ( wx.Frame ):
	def OnEraseBackground(self, evt):
		dc = evt.GetDC()
		if not dc:
			dc = wx.ClientDC(self)
			rect = self.GetUpdateRegion().GetBox()
			dc.SetClippingRect(rect)
		dc.Clear()
		bmp = wx.Bitmap("../../img/eye2.jpg")
		dc.DrawBitmap(bmp, 0, 0)
	def __init__( self, parent ):
		wx.Frame.__init__ ( self, parent, id = wx.ID_ANY, title = wx.EmptyString, pos = wx.DefaultPosition, size = wx.Size( 642,471 ), style = wx.DEFAULT_FRAME_STYLE|wx.TAB_TRAVERSAL )
		
		self.SetSizeHintsSz( wx.DefaultSize, wx.DefaultSize )
		
		bSizer2 = wx.BoxSizer( wx.VERTICAL )
		
		self.m_staticText1 = wx.StaticText( self, wx.ID_ANY, u"眼底图像分类系统", wx.DefaultPosition, wx.DefaultSize, wx.ALIGN_CENTRE )
		self.m_staticText1.Wrap( -1 )
		self.m_staticText1.SetFont( wx.Font( 22, 74, 90, 92, False, "微软雅黑" ) )
		
		bSizer2.Add( self.m_staticText1, 0, wx.ALIGN_CENTER_HORIZONTAL|wx.ALL, 20 )
		
		gSizer7 = wx.GridSizer( 0, 2, 0, 0 )
		
		self.m_staticText2 = wx.StaticText( self, wx.ID_ANY, u"训练集文件夹选择：", wx.DefaultPosition, wx.DefaultSize, wx.ALIGN_CENTRE )
		self.m_staticText2.Wrap( -1 )
		self.m_staticText2.SetFont( wx.Font( 16, 74, 90, 90, False, "微软雅黑" ) )
		
		gSizer7.Add( self.m_staticText2, 0, wx.ALIGN_RIGHT|wx.ALL, 20 )
		
		self.m_dirPicker1 = wx.DirPickerCtrl( self, wx.ID_ANY, wx.EmptyString, u"选择训练集文件夹", wx.DefaultPosition, wx.DefaultSize, wx.DIRP_DEFAULT_STYLE )
		gSizer7.Add( self.m_dirPicker1, 0, wx.ALIGN_LEFT|wx.ALL, 15 )
		
		self.m_staticText3 = wx.StaticText( self, wx.ID_ANY, u"测试集文件夹选择：", wx.DefaultPosition, wx.DefaultSize, wx.ALIGN_CENTRE )
		self.m_staticText3.Wrap( -1 )
		self.m_staticText3.SetFont( wx.Font( 16, 74, 90, 90, False, "微软雅黑" ) )
		
		gSizer7.Add( self.m_staticText3, 0, wx.ALIGN_RIGHT|wx.ALL, 20 )
		
		self.m_dirPicker2 = wx.DirPickerCtrl( self, wx.ID_ANY, wx.EmptyString, u"选择测试集文件夹 a folder", wx.DefaultPosition, wx.DefaultSize, wx.DIRP_DEFAULT_STYLE )
		gSizer7.Add( self.m_dirPicker2, 0, wx.ALL, 15 )
		
		
		bSizer2.Add( gSizer7, 1, wx.EXPAND, 5 )
		
		gSizer9 = wx.GridSizer( 0, 2, 0, 0 )
		
		self.m_staticText3 = wx.StaticText( self, wx.ID_ANY, u"功能选择：", wx.DefaultPosition, wx.DefaultSize, wx.ALIGN_CENTRE )
		self.m_staticText3.Wrap( -1 )
		self.m_staticText3.SetFont( wx.Font( 18, 74, 90, 92, False, "微软雅黑" ) )
		
		gSizer9.Add( self.m_staticText3, 0, wx.ALIGN_RIGHT|wx.ALIGN_TOP|wx.ALL, 50 )
		
		bSizer3 = wx.BoxSizer( wx.VERTICAL )
		
		self.m_button1 = wx.Button( self, wx.ID_ANY, u"提取特征", wx.DefaultPosition, wx.DefaultSize, 0 )
		bSizer3.Add( self.m_button1, 0, wx.ALIGN_CENTER|wx.ALL, 10)
		
		self.m_button2 = wx.Button( self, wx.ID_ANY, u"训练网络", wx.DefaultPosition, wx.DefaultSize, 0 )
		bSizer3.Add( self.m_button2, 0, wx.ALIGN_CENTER|wx.ALL, 10)
		
		
		self.m_button3 = wx.Button( self, wx.ID_ANY, u"模型评估", wx.DefaultPosition, wx.DefaultSize, 0 )
		bSizer3.Add( self.m_button3, 0, wx.ALIGN_CENTER_HORIZONTAL|wx.ALL, 10 )
		# self.m_button4 = wx.Button( self, wx.ID_ANY, u"图片识别", wx.DefaultPosition, wx.DefaultSize, 0 )
		# bSizer3.Add( self.m_button4, 0, wx.ALIGN_CENTER_HORIZONTAL|wx.ALL, 5 )
		
		gSizer9.Add( bSizer3, 1, wx.EXPAND, 5 )
		
		self.m_staticText4 = wx.StaticText( self, wx.ID_ANY, u"注：使用模型评估功能前必须先训练网络", wx.DefaultPosition, wx.DefaultSize, 0 )
		self.m_staticText4.Wrap( -1 )
		self.m_staticText4.SetFont( wx.Font( 11, 75, 90, 90, False, "新宋体" ) )
		
		gSizer9.Add( self.m_staticText4, 0, wx.ALIGN_CENTER|wx.ALL, 20 )
		self.Bind(wx.EVT_ERASE_BACKGROUND, self.OnEraseBackground)
		
		bSizer2.Add( gSizer9, 1, wx.EXPAND, 5 )
		
		
		self.SetSizer( bSizer2 )
		self.Layout()
		
		self.Centre( wx.BOTH )
		
		# Connect Events
		self.m_button1.Bind( wx.EVT_LEFT_DOWN, self.m_button1OnLeftClick )
		self.m_button2.Bind( wx.EVT_LEFT_DOWN, self.m_button2OnLeftClick )
		self.m_button3.Bind( wx.EVT_LEFT_DOWN, self.m_button3OnLeftClick )
		#self.m_button4.Bind( wx.EVT_LEFT_DOWN, self.m_button4OnLeftClick )
		self.train_data_dir = ''
		self.test_data_dir = ''
		# self.m_button2.Disable()
		self.signal = 0
	def __del__( self ):
		pass
	
	
	# Virtual event handlers, overide them in your derived class
	def m_button1OnLeftClick( self, event ):
		self.train_data_dir = self.m_dirPicker1.GetPath()
		self.test_data_dir = self.m_dirPicker2.GetPath()
		if self.train_data_dir =='':
			ret = wx.MessageBox(u'未选取训练集文件夹！',  'Alert', wx.OK)
			
		elif self.test_data_dir =='':
			ret = wx.MessageBox(u'未选取验证集文件夹！',  'Alert', wx.OK)
			
		else:
			self.signal = 1
			
			thirdframe.ThirdFrame(self).Show()
			self.Hide()
		event.Skip()
	
	def m_button2OnLeftClick( self, event ):
		self.train_data_dir = self.m_dirPicker1.GetPath()
		self.test_data_dir = self.m_dirPicker2.GetPath()
			
		if os.path.exists('npy/bottleneck_features_train.npy') == False or os.path.exists('npy/bottleneck_features_validation.npy')==False:
			ret = wx.MessageBox(u'未发现特征文件，请先提取特征！',  'Alert', wx.OK)
		elif self.train_data_dir =='':
			ret = wx.MessageBox(u'未选取训练集文件夹！',  'Alert', wx.OK)
			
		elif self.test_data_dir =='':
			ret = wx.MessageBox(u'未选取验证集文件夹！',  'Alert', wx.OK)
		else:
			self.signal = 2
			secondframe.SecondFrame(self).Show()
			self.Hide()

		event.Skip()
	
	def m_button3OnLeftClick( self, event ):
		if os.path.exists('softmax.h5') == False:
			ret = wx.MessageBox(u'未发现网络参数文件，请先训练网络！',  'Alert', wx.OK)
		else:
			self.signal = 3
			thirdframe.ThirdFrame(self).Show()
			self.Hide()
		event.Skip()
	# def m_button4OnLeftClick(self, event):
		
	# 	pickpic.Pick(self).Show()
	# 	self.Hide()
	# 	event.Skip()
if __name__ == '__main__':  
    app = wx.App()  
    MyFrame1(None).Show()
    app.MainLoop()   
		
		

	
 
