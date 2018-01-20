# -*- coding: utf-8 -*- 

###########################################################################
## Python code generated with wxFormBuilder (version Aug 23 2015)
## http://www.wxformbuilder.org/
##
## PLEASE DO "NOT" EDIT THIS FILE!
###########################################################################
from sys import path
path.append(r'../core')
import wx
import wx.xrc
import wx.richtext
import train
import thread
import sys
###########################################################################
## Class MyFrame2
###########################################################################
class RedirectText(wx.TextCtrl):
	def __init__(self, parent, id, title,pos,size):
		wx.TextCtrl.__init__(self,parent, id, title,pos,size,style=wx.TE_MULTILINE)
		self.old_stdout=sys.stdout
		
	def flush(self):   
		self.buff=''   
	def write(self, string):
		wx.CallAfter(self.WriteText, string)
		self.flush()

class ThirdFrame ( wx.Frame ):
	

	
	def __init__( self, parent ):
		wx.Frame.__init__ ( self, parent, id = wx.ID_ANY, title = wx.EmptyString, pos = wx.DefaultPosition, size = wx.Size( 515,390 ), style = wx.DEFAULT_FRAME_STYLE|wx.TAB_TRAVERSAL )
		
		self.SetSizeHintsSz( wx.DefaultSize, wx.DefaultSize )
		
		bSizer2 = wx.BoxSizer( wx.VERTICAL )
		
		self.m_staticText1 = wx.StaticText( self, wx.ID_ANY, u"训练过程", wx.DefaultPosition, wx.DefaultSize, 0 )
		self.m_staticText1.Wrap( -1 )
		self.m_staticText1.SetFont( wx.Font( 15, 70, 90, 90, False, "Lucida Grande" ) )
		
		bSizer2.Add( self.m_staticText1, 0, wx.ALIGN_CENTER_HORIZONTAL|wx.ALL, 10 )
		
		self.m_richText1 = RedirectText( self, wx.ID_ANY, wx.EmptyString, wx.DefaultPosition, wx.DefaultSize)
		sys.stdout=self.m_richText1
		
		
		bSizer2.Add( self.m_richText1, 1, wx.EXPAND |wx.ALL, 5 )
		
		self.m_button1 = wx.Button( self, wx.ID_ANY, u"返回主页面", wx.DefaultPosition, wx.DefaultSize, 0 )
		self.m_button1.Disable()
		bSizer2.Add( self.m_button1, 0, wx.ALIGN_RIGHT|wx.ALL, 5 )
		
		
		self.SetSizer( bSizer2 )
		self.Layout()
		
		self.Centre( wx.BOTH )
		self.parent = parent
		self.train_data_dir = parent.train_data_dir
		self.test_data_dir = parent.test_data_dir
		if parent.signal == 2:
			self.learnRate = parent.learnRate
			self.optimizerType = parent.optimizerType
			self.epoch_num = parent.epoch_num
		# Connect Events
		self.m_button1.Bind( wx.EVT_LEFT_DOWN, self.m_button1OnLeftDClick )
		#self.m_richText1.WriteText()
		self.Bind(wx.EVT_CLOSE,self.OnClose)
		self.train()
	def train(self):
		if self.parent.signal == 1:
			train_model = train.TrainModel(self.train_data_dir,self.test_data_dir,'1','1','1',self.m_richText1,self.m_button1)
			thread.start_new_thread(train_model.save_bottlebeck_features,())
		elif self.parent.signal == 2:
			train_model = train.TrainModel(self.train_data_dir,self.test_data_dir,self.optimizerType,self.learnRate,self.epoch_num,self.m_richText1,self.m_button1)
			thread.start_new_thread(train_model.train_top_model,())
		elif self.parent.signal == 3:
			train_model = train.TrainModel('1','1','1','1','1',self.m_richText1,self.m_button1)
			thread.start_new_thread(train_model.predict,())


	def __del__( self ):
		self.parent.Destroy()
		thread.exit_thread()
	
	# Virtual event handlers, overide them in your derived class
	def m_button1OnLeftDClick( self, event ):
		self.Hide()
		if self.parent.signal == 2:
			self.parent.parent.m_button2.Enable()
			
			self.parent.parent.Show()
		else:
			self.parent.m_button2.Enable()

			self.parent.Show()
		event.Skip()
	
	def OnClose(self, evt):
		ret = wx.MessageBox(u'程序运行中，确定要退出吗？',  'Alert', wx.OK|wx.CANCEL)
		if ret == wx.OK:
			thread.exit_thread()
			evt.Skip()
		
