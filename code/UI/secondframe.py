# -*- coding: utf-8 -*- 

###########################################################################
## Python code generated with wxFormBuilder (version Aug 23 2015)
## http://www.wxformbuilder.org/
##
## PLEASE DO "NOT" EDIT THIS FILE!
###########################################################################

import wx
import wx.xrc
import thirdframe
###########################################################################
## Class SechondFrame
###########################################################################

class SecondFrame ( wx.Frame ):
	def OnEraseBackground(self, evt):
		dc = evt.GetDC()
		if not dc:
			dc = wx.ClientDC(self)
			rect = self.GetUpdateRegion().GetBox()
			dc.SetClippingRect(rect)
		dc.Clear()
		bmp = wx.Bitmap("../../img/background2.jpg")
		dc.DrawBitmap(bmp, 0, 0)
	
	def __init__( self, parent ):
		wx.Frame.__init__ ( self, parent, id = wx.ID_ANY, title = u"参数设置", pos = wx.DefaultPosition, size = wx.Size( 490,285 ), style = wx.DEFAULT_FRAME_STYLE|wx.TAB_TRAVERSAL )
		
		self.SetSizeHintsSz( wx.DefaultSize, wx.DefaultSize )
		
		bSizer1 = wx.BoxSizer( wx.VERTICAL )
		
		self.m_staticText1 = wx.StaticText( self, wx.ID_ANY, u"参数设置", wx.DefaultPosition, wx.DefaultSize, 0 )
		self.m_staticText1.SetForegroundColour("Red")
		self.m_staticText1.Wrap( -1 )
		self.m_staticText1.SetFont( wx.Font( 15, 70, 90, 91, False, "Heiti SC" ) )
		
		bSizer1.Add( self.m_staticText1, 0, wx.ALIGN_CENTER_HORIZONTAL|wx.ALL, 15 )
		
		gSizer1 = wx.GridSizer( 0, 2, 0, 0 )
		
		self.m_staticText2 = wx.StaticText( self, wx.ID_ANY, u"优化器选择：", wx.DefaultPosition, wx.DefaultSize,0 )
		self.m_staticText2.SetForegroundColour("Red")
		self.m_staticText2.Wrap( -1 )
		gSizer1.Add( self.m_staticText2, 0, wx.ALIGN_RIGHT|wx.ALL, 10 )
		
		self.m_choice1Choices = [ u"Adam", u"SGD" ]
		self.m_choice1 = wx.Choice( self, wx.ID_ANY, wx.DefaultPosition, wx.DefaultSize, self.m_choice1Choices, 0 )
		self.m_choice1.SetSelection( 0 )
		# self.Bind(wx.EVT_CHOICE, self.chooseScoreFunc, self.m_choice1)
		gSizer1.Add( self.m_choice1, 0, wx.ALL, 10 )
		
		self.m_staticText3 = wx.StaticText( self, wx.ID_ANY, u"学习率设置：", wx.DefaultPosition, wx.DefaultSize, 0 )
		self.m_staticText3.SetForegroundColour("Red")
		self.m_staticText3.Wrap( -1 )
		gSizer1.Add( self.m_staticText3, 0, wx.ALIGN_RIGHT|wx.ALL, 10 )
		
		self.m_textCtrl1 = wx.TextCtrl( self, wx.ID_ANY, u"1e-6", wx.DefaultPosition, wx.DefaultSize, 0 )
		gSizer1.Add( self.m_textCtrl1, 0, wx.ALL, 10 )
		
		self.m_staticText4 = wx.StaticText( self, wx.ID_ANY, u"训练周期数：", wx.DefaultPosition, wx.DefaultSize, 0 )
		self.m_staticText4.SetForegroundColour("Red")
		self.m_staticText4.Wrap( -1 )
		gSizer1.Add( self.m_staticText4, 0, wx.ALIGN_RIGHT|wx.ALL, 10 )
		self.m_textCtrl2 = wx.TextCtrl( self, wx.ID_ANY, u"50", wx.DefaultPosition, wx.DefaultSize, 0 )
		gSizer1.Add( self.m_textCtrl2, 0, wx.ALL, 10 )
		
		gSizer1.AddSpacer( ( 0, 0), 1, wx.EXPAND, 5 )
		
		self.m_button1 = wx.Button( self, wx.ID_ANY, u"确定", wx.DefaultPosition, wx.DefaultSize, 0 )
		gSizer1.Add( self.m_button1, 0, wx.ALL, 10 )
		
		
		bSizer1.Add( gSizer1, 1, wx.EXPAND, 5 )
		
		self.m_staticText4 = wx.StaticText( self, wx.ID_ANY, u"注：学习率格式为0.001 或1e-6", wx.DefaultPosition, wx.DefaultSize, 0 )
		self.m_staticText4.SetForegroundColour("Red")
		self.m_staticText4.Wrap( -1 )
		self.m_staticText4.SetFont( wx.Font( 13, 70, 90, 92, False, "Lucida Grande" ) )
		
		bSizer1.Add( self.m_staticText4, 0, wx.ALIGN_CENTER_HORIZONTAL|wx.ALL, 5 )
		self.Bind(wx.EVT_ERASE_BACKGROUND, self.OnEraseBackground)
		
		bSizer1.AddSpacer( ( 0, 0), 1, wx.EXPAND, 5 )
		self.signal = parent.signal
		
		self.SetSizer( bSizer1 )
		self.Layout()
		
		self.Centre( wx.BOTH )
		self.parent = parent
		self.train_data_dir = parent.train_data_dir
		self.test_data_dir = parent.test_data_dir
		
		# Connect Events
		self.m_button1.Bind( wx.EVT_LEFT_DOWN, self.m_button1OnLeftDClick )
	
	def __del__( self ):
		self.parent.Destroy()
	
	
	# Virtual event handlers, overide them in your derived class
	def m_button1OnLeftDClick( self, event ):
		self.learnRate = self.m_textCtrl1.GetValue()
		self.epoch_num = self.m_textCtrl2.GetValue()
		self.optimizerType = self.m_choice1Choices[int(self.m_choice1.GetSelection())]
		thirdframe.ThirdFrame(self).Show()
		self.Hide()
		event.Skip()
	# def chooseScoreFunc(self,event):
	# 	self.optimizerType = str(event.GetEventObject().GetSelection())


