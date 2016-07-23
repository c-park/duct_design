
"""
File: data-gui.py
Author: Cade Parkison
Email: cadeparkison@gmail.com
Github: c-park

Description: Data Widgets for Duct System Design

"""

from IPython.display import display, clear_output
from ipywidgets import widgets
import numpy as np
import pandas as pd

def input_df(self):
    # Creates a dataframe of input data
    input_columns = ["$S$", "$S_1$", "$S_2$", "rect", "presized", "$Q$",
                     "$L$", "$\Delta P_z$", "$e$", "$H_z$", "$W_z$",
                     "$D_z$", "$C$"]
    self.systemDF = pd.DataFrame(columns=input_columns)
    for (i, s) in self.sections.items():
        sectionDF = pd.DataFrame(
            data=[
                [s._id, s.ch1, s.ch2, s.is_rect, s.is_presized, s.air_flow,
                    s.duct_length, s.delta_p_z, s.rough_factor, s.req_height,
                    s.req_width, s.req_dia, s.loss_coeff]],
            columns=input_columns, index=[s._id])
        self.systemDF[
            ["$S$", "$S_1$", "$S_2$"]] = self.systemDF[
            ["$S$", "$S_1$", "$S_2$"]].astype(int)
        self.systemDF = self.systemDF.append(sectionDF)

    self.systemDF = np.round(self.systemDF.set_index("$S$"), 3)

    return self.systemDF


def vel_size_fric_df(self):
    # Creates a dataframe of velocity, size, and friction data
    vsf_columns = ["$S$", "$V$", "$H$", "$W$", "$D_f$", "$D_v$", "$f$",
                   "$\Delta P$"]
    self.systemDF = pd.DataFrame(columns=vsf_columns)

    for (i, s) in self.sections.items():
        sectionDF = pd.DataFrame(
            data=[[s._id, s.velocity, s.height, s.width,
                   s.dia_by_frict, s.dia_by_vel, s.f_factor,
                   s.total_delta_P]],
            columns=vsf_columns,
            index=[int(s._id)])
        self.systemDF[["$S$"]] = self.systemDF[["$S$"]].astype(int)
        self.systemDF = self.systemDF.append(sectionDF)

    self.systemDF = np.round(self.systemDF.set_index("$S$"), 3)

    return self.systemDF


def condense_expand_df(self):
    # Creates a dataframe of condensing, expansion, pressure loss data
    condense_expand_columns = ["$S$", "$\mu$", "$K_s$", "$K_t$", "$T$",
                               "$\Delta P_{max}$"]
    self.systemDF = pd.DataFrame(columns=condense_expand_columns)
    for s in self.sections.items():
        sectionDF = pd.DataFrame(
            data=[[s._id, s.mu, s.Ks, s.Kt, s.T, s.max_delta_P]],
            columns=condense_expand_columns,
            index=[s._id])
        self.systemDF[["$S$"]] = self.systemDF[["$S$"]].astype(int)
        self.systemDF = self.systemDF.append(sectionDF)

    self.systemDF = np.round(self.systemDF.set_index("$S$"), 3)

    return self.systemDF


def display_data(self):
    """ displays all data in pandas dataframes
    :returns: none

    """
    input_data = self.input_df()
    vel_size_fric_data = self.vel_size_fric_df()
    condense_expand_data = self.condense_expand_df()

    display(input_data)
    display(vel_size_fric_data)
    display(condense_expand_data)


def myDropdown(options=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
               text='',
               width='50px'):
    c1 = widgets.VBox()
    drop = widgets.Dropdown(options=options, width=width)
    text = widgets.Label(text)
    c1.children = (text, drop)
    return drop, c1


def myFloatText(text='', width='50px'):
    c1 = widgets.VBox()
    floats = widgets.FloatText(width=width)
    text = widgets.Label(text)
    c1.children = (text, floats)
    return floats, c1


def myCheckbox(value=False, text='', width='50px'):
    c1 = widgets.HBox()
    check = widgets.Checkbox(value=value, width=width)
    text = widgets.Label(text)
    c1.children = (text, check)
    return check, c1


def display_form():
    id_, id_drop = myDropdown(text='$S$')
    ch1_, ch1_drop = myDropdown(text='$S_1$')
    ch2_, ch2_drop = myDropdown(text='$S_2$')
    flow_, flow_text = myFloatText(text='$Q$')
    length_, length_text = myFloatText(text='$L$')
    p_loss_, p_loss_text = myFloatText(text='$\Delta P_z$')
    roughness_, roughness_text = myFloatText(text='$e$')
    rect_, rect_checkbox = myCheckbox(text='Rectangle')
    presized_, presized_checkbox = myCheckbox(text='Presized')
    height_, height_text = myFloatText(text='$H_z$')
    width_, width_text = myFloatText(text='$W_z$')
    diameter_, diameter_text = myFloatText(text='$D_z$')
    loss_, loss_text = myFloatText(text='$C$')

    export_fName_text = widgets.Text(description="File Name:")

    add_button = widgets.Button(description="Add Section")
    edit_button = widgets.Button(description="Edit Section")
    delete_button = widgets.Button(description="Delete Section")
    export_system_button = widgets.Button(description="Export Data")
    new_system_button = widgets.Button(description='New System')

    export_system_form = widgets.VBox(
        children=[export_system_button, export_fName_text])

    save_system_form = widgets.VBox(children=[new_system_button])

    button_container = widgets.HBox(
        children=[
            add_button,
            edit_button,
            delete_button,
            rect_checkbox,
            presized_checkbox,
            export_system_form])
    form_container = widgets.HBox(
        children=[
            id_drop,
            ch1_drop,
            ch2_drop,
            flow_text,
            length_text,
            p_loss_text,
            roughness_text,
            height_text,
            width_text,
            diameter_text,
            loss_text,
            save_system_form])

    form = widgets.VBox(children=[button_container, form_container])

    return form


def on_add_button_clicked(b):
    section = DuctSection(
        _id=id_.value,
        ch1=ch1_.value,
        ch2=ch2_.value,
        is_presized=presized_.value,
        is_rect=rect_.value,
        air_flow=flow_.value,
        duct_length=length_.value,
        delta_p_z=p_loss_.value,
        rough_factor=roughness_.value,
        req_height=height_.value,
        req_width=width_.value,
        req_dia=diameter_.value,
        loss_coeff=loss_.value,
        f_factor=0.019)
    System.add_section(section)
    clear_output()
    display(np.round(System.input_df(), 3))


def on_edit_button_clicked(b):
    print(1)


def on_delete_button_clicked(b):
    pass


def on_export_button_clicked(b):
    System.export_system(export_fName_text.value)


def on_new_system_button_clicked(b):
    clear_output()
    global System
    System = DuctSystem()
    display(np.round(System.input_df(), 3))
