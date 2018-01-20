# -*- coding: utf-8 -*-

###########################################################################
# Python code generated with wxFormBuilder (version Aug 23 2015)
# http://www.wxformbuilder.org/
##
# PLEASE DO "NOT" EDIT THIS FILE!
###########################################################################

import wx
import wx.xrc
import wx.richtext
import sys
import train
import thread

###########################################################################
# Class fourthframe
###########################################################################


class RedirectText(wx.TextCtrl):

    def __init__(self, parent, id, title, pos, size):
        wx.TextCtrl.__init__(self, parent, id, title, pos,
                             size, style=wx.TE_MULTILINE)
        self.old_stdout = sys.stdout

    def flush(self):
        self.buff = ''

    def write(self, string):
        wx.CallAfter(self.WriteText, string)
        self.flush()


class fourthframe (wx.Frame):

    def __init__(self, parent):
        wx.Frame.__init__(self, parent, id=wx.ID_ANY, title=u"眼底图片识别", pos=wx.DefaultPosition, size=wx.Size(
            517, 386), style=wx.DEFAULT_FRAME_STYLE | wx.TAB_TRAVERSAL)

        self.SetSizeHintsSz(wx.DefaultSize, wx.DefaultSize)

        bSizer1 = wx.BoxSizer(wx.VERTICAL)
        self.picPath = parent.picPath
        self.m_bitmap1 = wx.StaticBitmap(
            self, wx.ID_ANY, wx.NullBitmap, wx.DefaultPosition, (80, 80), 0)
        img = wx.Image(self.picPath, wx.BITMAP_TYPE_ANY).Scale(80, 80)
        self.m_bitmap1.SetBitmap(wx.BitmapFromImage(img))
        bSizer1.Add(self.m_bitmap1, 0, wx.ALIGN_CENTER_HORIZONTAL | wx.ALL, 20)

        self.m_richText1 = RedirectText(
            self, wx.ID_ANY, wx.EmptyString, wx.DefaultPosition, wx.DefaultSize)
        sys.stdout = self.m_richText1
        bSizer1.Add(self.m_richText1, 1, wx.EXPAND | wx.ALL, 5)

        self.m_button1 = wx.Button(
            self, wx.ID_ANY, u"返回主界面", wx.DefaultPosition, wx.DefaultSize, 0)
        bSizer1.Add(self.m_button1, 0, wx.ALIGN_RIGHT | wx.ALL, 5)

        self.Bind(wx.EVT_CLOSE, self.OnClose)
        self.SetSizer(bSizer1)
        self.Layout()

        self.Centre(wx.BOTH)
        self.parent = parent
        # Connect Events
        self.m_button1.Bind(wx.EVT_BUTTON, self.m_button1OnButtonClick)
        self.m_button1.Disable()
        self.train()

    def __del__(self):
        pass

    def train(self):
        train_model = train.TrainModel(
            '1', '1', '1', '1', self.m_richText1, self.m_button1)
        thread.start_new_thread(train_model.predict2, (self.picPath,))

    # Virtual event handlers, overide them in your derived class
    def m_button1OnButtonClick(self, event):
        self.parent.parent.Show()
        event.Skip()

    def OnClose(self, evt):
        ret = wx.MessageBox(u'程序运行中，确定要退出吗？',  'Alert', wx.OK | wx.CANCEL)
        if ret == wx.OK:
            thread.exit_thread()
            evt.Skip()
if __name__ == '__main__':
    app = wx.App()
    fourthframe(None).Show()
    app.MainLoop()
