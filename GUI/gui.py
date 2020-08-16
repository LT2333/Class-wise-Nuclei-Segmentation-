import shutil

import kivy
from kivy.app import App
from kivy.core.window import Window
from kivy.properties import ObjectProperty
from kivy.uix.boxlayout import BoxLayout
import argparse

from kivy.uix.gridlayout import GridLayout
from kivy.uix.label import Label
from kivy.uix.button import Button
from kivy.uix.image import Image
from kivy.uix.floatlayout import FloatLayout
from kivy.uix.popup import Popup
import gui_utils

kivy.require('1.10.1')
Window.size = (1000, 600)


class MainScreen(BoxLayout, FloatLayout):
    def dismiss_popup(self):
        self._popup.dismiss()

    def show_load(self):
        content = LoadDialog(load=self.load, cancel=self.dismiss_popup)
        self._popup = Popup(title="Select Image", content=content,
                            size_hint=(0.9, 0.9))
        self._popup.open()

    def load(self, path, filename):
        self.ids.img.source = filename[0]

        # sentences = sample.start_app_pred(self.ids.img.source, args.encoder_path, args.decoder_path)
        # self.ids.pred.text = "Prediction: xxx"


class LoadDialog(BoxLayout, FloatLayout):
    load = ObjectProperty(None)
    cancel = ObjectProperty(None)
    def selected(self, filename):
        ImgPred().selected(filename)


class ImgPred(BoxLayout, FloatLayout):
    load = ObjectProperty(None)
    cancel = ObjectProperty(None)

    def selected(self, filename):
        float = FloatLayout()

        input_img = filename[0]

        float.add_widget(Label(text='Input Image', pos_hint={'top': 1.42, 'right': 0.65}, font_size='20sp'))
        float.add_widget(Label(text='Segmentation', pos_hint={'top': 1.42, 'right': 1.0}, font_size='20sp'))
        float.add_widget(Label(text='Classification', pos_hint={'top': 1.42, 'right': 1.35}, font_size='20sp'))

        float.add_widget(Image(source=input_img, pos_hint={'top': 1.15, 'right': 0.65}))

        output_imgs = gui_utils.model_start(input_img)

        overlay = output_imgs[0]
        float.add_widget(Image(source=overlay, pos_hint={'top': 1.15, 'right': 1.0}))

        output_img_cls = output_imgs[1]
        float.add_widget(Image(source=output_img_cls, pos_hint={'top': 1.15, 'right': 1.35}))

        class_labels = 'class_labels.png'
        float.add_widget(Image(source=class_labels, size_hint=(0.6, 0.6), pos_hint={'bottom': 0.0, 'right': 0.8}))

        btn = Button(text='Click to Select Another Image', size_hint=(1, 0.15), font_size=20, pos_hint={'bottom': 0.1, 'right': 1})
        btn.bind(on_press=self.show_load)
        float.add_widget(btn)

        gui_utils.remove_temp_dir(input_img)

        self._popup = Popup(content=float, title="SOA -- Nuclei Segmentation")
        self._popup.open()

    def dismiss_popup(self):
        self._popup.dismiss()

    def show_load(self, *args):
        content = LoadDialog(load=self.load, cancel=self.dismiss_popup)
        self._popup = Popup(title="Select Image", content=content,
                            size_hint=(0.9, 0.9))
        self._popup.open()


class NucleiSegmentationApp(App):
    kv_directory = 'kivy_utils'
    def build(self):
        return MainScreen()


if __name__ == '__main__':
    NucleiSegmentationApp().run()



