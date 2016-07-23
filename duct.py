"""
File: duct.py
Author: Cade Parkison
Email: cadeparkison@gmail.com
Github:

Description: test

"""

import numpy as np

import csv


# air temperature (deg C)
temp = 22
# absolute roughness  (m)
e = 0.0003
# kinematic viscosity (m^2/s)
kVisc = 1.54*10**(-5)
# air density        (kg/m^3)
density = 1.2
# fan efficiency
efficiency_fan_peak = 0.85
efficiency_fan_op = 0.75
# motor efficiency
efficiency_motor = 0.8
# total system airflow  (m^3/s)
Qfan = 1.42

#  Economic Data
energy_cost = 2.03      # (c/kWh)
duct_cost = 43.27       # ($/m^2)
sys_oper_time = 4400    # (h/yr)


class DuctSection:

    def __init__(self, **kwargs):
        """TODO:

        :**kwargs:

        """

        prop_defaults = {
            "_id": 0,
            "ch1": 0,
            "ch2": 0,
            "is_presized": False,
            "is_rect": False,
            "air_flow": 0,
            "duct_length": 1,
            "delta_p_z": 1,
            "rough_factor": 0,
            "roughness": 0.0003,
            "req_height": 0,
            "req_width": 0,
            "req_dia": 0,
            "loss_coeff": 1,
            "f_factor": 0.019,
            "kVisco": 1.54*10**(-5)
        }

        for (prop, default) in prop_defaults.items():
            setattr(self, prop, kwargs.get(prop, default))

        global density
        self.is_terminal = self.ch1 == 0 & self.ch2 == 0

    def data_init(self):
        """
        Data Initiation
        """

        if self.is_presized:
            # presized rectangular duct
            if self.is_rect:
                self.height = self.req_height
                self.width = self.req_width
                self.velocity = self.get_velocity()
                self.dia_by_frict = 2*(self.height*self.width)/(
                    self.height+self.width)
                self.dia_by_vel = 1.128 * (self.air_flow / self.velocity)**0.5
            # presized round duct
            else:
                self.height = 0
                self.width = 0
                self.dia_by_vel = self.req_dia
                self.dia_by_frict = self.req_dia
                self.velocity = self.get_velocity()
        else:
            self.velocity = 6.5
            self.dia_by_vel = 1.128 * (self.air_flow / self.velocity)**0.5
            # rectangular duct
            if self.is_rect:
                self.height = self.req_height
                self.width = self.air_flow / (self.velocity * self.height)
                self.dia_by_frict = 2*(self.height*self.width)/(
                    self.height+self.width)
            # round duct
            else:
                self.height = 0
                self.width = 0
                self.dia_by_frict = self.dia_by_vel

        self.total_delta_P = self.get_total_delta_P()

    def iterate1(self):
        self.dia_by_vel = self.est_dia
        if self.is_rect:
            self.width = 0.785 * self.dia_by_vel**2 / self.height
        if not self.is_presized:
            self.velocity = self.get_velocity()
        self.f_factor = self.get_f_factor()
        self.total_delta_P = self.get_total_delta_P()

    def update_height(self):
        """
        TODO: INCOMPLETE
        """
        if self.is_rect:
            self.height = self.req_height
        else:
            self.height = 0

    def get_system(self):
        pass

    def get_velocity(self):
        if self.is_rect:
            velocity = self.air_flow / (self.height*self.width)
        else:
            velocity = (1.274*self.air_flow) / (self.dia_by_vel**2)
        return velocity

    def get_cross_section(self):
        pass

    def update_dia_by_frict(self):

        if self.is_rect:
            self.dia_by_frict = 2*(self.height*self.width) / (self.height
                                                              + self.width)
        else:
            if self.is_presized:
                self.dia_by_frict = self.req_dia
            else:
                self.dia_by_frict = self.dia_by_vel

    def get_f_factor(self):
        self.update_dia_by_frict()
        if self.is_rect:
            dia = self.dia_by_frict
        else:
            dia = self.dia_by_vel

        re = dia * self.velocity / self.kVisco
        f = 0.11*(self.roughness/self.dia_by_frict + 68/re)**0.25
        return f

    def get_total_delta_P(self):
        total_delta_P = ((self.f_factor*self.duct_length / self.dia_by_frict
                          + self.loss_coeff) * density*self.velocity**2/2
                         + self.delta_p_z)
        return total_delta_P

    def update_mu(self):
        if self.is_rect:
            mu = (self.f_factor*self.duct_length/self.dia_by_frict
                  + self.loss_coeff)*self.dia_by_vel
        else:
            mu = (self.f_factor*self.duct_length
                  + self.loss_coeff*self.dia_by_vel)
        self.mu = mu

    def update_Ks(self):
        if self.is_presized:
            Ks = 0
        else:
            if self.is_rect:
                r = self.height / self.width
                n = (1 + r)/(np.sqrt(np.pi*r))
            else:
                n = 1
            Ks = n*(self.mu**0.2)*(self.air_flow**0.4)*self.duct_length
        self.Ks = Ks

    def update_Kt(self, system):

        if self.is_terminal:
            Kt = self.Ks
        else:
            child1 = system.get_section(self.ch1)
            child2 = system.get_section(self.ch2)

            if child1.is_presized:
                K_1 = child1.Kt
            else:
                K_1 = child1.Ks

            if child2.is_presized:
                K_2 = child2.Kt
            else:
                K_2 = child2.Ks

            # equation 1.41
            Kt = ((K_1 + K_2)**0.833 + self.Ks**0.833)**1.2
        self.Kt = Kt

    def update_T(self):

        if self.is_terminal:
            T = 1
        else:
            if self.is_presized:
                T = 0
            else:
                T = (self.Ks / self.Kt)**0.833
        self.T = T

    def update_max_delta_P(self, system):
        if self.is_terminal:
            if self.is_presized:
                self.max_delta_P = self.total_delta_P
            else:
                self.max_delta_P = self.delta_p_z
        else:
            if self.is_presized:
                delta_p_z = self.total_delta_P
            else:
                delta_p_z = self.delta_p_z
            child1 = system.get_section(self.ch1)
            child2 = system.get_section(self.ch2)
            self.max_delta_P = max(child1.max_delta_P,
                                   child2.max_delta_P)  \
                + delta_p_z

    def get_section_coeff(self):
        pass


class DuctSystem:

    def __init__(self, **kwargs):

        prop_defaults = {
            # General Data
            "temp": 22,                   # degrees Celsius
            "roughness": 0.0003,          # meters
            "kVisco": 1.54*10**(-5),    # m^2/sec
            "density": 1.2,               # kg/m^3
            "efficiency_fan_peak": 0.85,
            "efficiency_fan_op": 0.75,
            "efficiency_motor": 0.8,
            "total_airflow": 1.42,        # m^3/sec
            # Economic Data
            "energy_cost": 2.03,          # c/kWh
            "duct_cost": 43.27,           # $/m^2
            "oper_time": 4400,             # h/yr
            "PWEF": 8.61,                  # present worth escalation factor
            "dim_const": 1.0           # dimensional constant (kg-m)/(N-s^2)
        }

        for (prop, default) in prop_defaults.items():
            setattr(self, prop, kwargs.get(prop, default))

        self.sections = {}
        self.parents = self.find_parents()

    def add_section(self, _id, section):
        self.sections[_id] = section

    def get_section(self, _id):
        # return DuctSection instance
        if _id in self.sections.keys():
            section = self.sections[_id]
        else:
            section = 0
        return section

    def data_init(self):
        for (i, section) in self.sections.items():
            section.data_init()
            if i not in self.parents.keys():
                self.root = section
        self.condense()
        self.update_opt_fan_press()
        self.expand()
        self.update_path_delta_P()

    def iterate(self):
        for (i, section) in self.sections.items():
            section.iterate1()
        self.condense()
        self.update_opt_fan_press()
        self.expand()
        self.update_path_delta_P()

    def find_parents(self):
        """ Crete parents dictionary of (section: parent) pairs
        """
        parents = {}

        for (i, section) in self.sections.items():
            child1_id = section.ch1
            child2_id = section.ch2
            for child in [child1_id, child2_id]:
                if child != 0:
                    parents[child] = section._id
        return parents

    def condense(self):
        """
        System Condensing
        """
        for i in sorted(self.sections.keys()):
            section = self.sections[i]
            section.update_mu()
            section.update_Ks()
            section.update_Kt(self)
            section.update_T()
            section.update_max_delta_P(self)

    def expand(self):
        for i in sorted(self.sections.keys(), reverse=True):
            section = self.sections[i]
            if i != self.root._id:
                parent = self.get_section(self.parents[i])
            if section._id not in self.parents.keys():
                section.up_press = self.opt_fan_press
            else:
                section.up_press = parent.down_press

            if section.is_presized:
                section.req_delta_P = section.total_delta_P
                section.req_delta_P_total = section.req_delta_P
            else:
                section.req_delta_P = (section.up_press - section.max_delta_P
                                       ) * section.T
                section.req_delta_P_total = (section.req_delta_P
                                             + section.delta_p_z)
            section.down_press = (section.up_press
                                  - section.req_delta_P_total)
            if section.is_presized:
                section.est_dia = section.req_dia
            else:
                # print(section.mu, self.density, self.total_airflow
                section.est_dia = (
                    0.959 * section.mu ** 0.2 * self.density ** 0.2 *
                    section.air_flow ** 0.4
                    * (self.dim_const * section.req_delta_P) ** -0.2)

    def update_path_delta_P(self):

        def get_total_path_delta_P(section):
            # print("sec={} root={}".format(section._id, self.root._id))
            if section._id == self.root._id:
                return section.total_delta_P
            else:
                parent = self.get_section(self.parents[section._id])
                return section.total_delta_P + get_total_path_delta_P(parent)

        for (i, section) in self.sections.items():
            section.total_path_delta_P = get_total_path_delta_P(section)
            if section.is_terminal:
                section.exces_path_delta_P = (self.opt_fan_press
                                              - section.total_path_delta_P)
            else:
                section.exces_path_delta_P = None

    def update_max_delta_P(self):
        for (i, section) in self.sections.items():
            if section.is_terminal:
                if section.is_presized:
                    section.max_delta_P = section.total_delta_P
                else:
                    section.max_delta_P = section.delta_p_z
            else:
                if section.is_presized:
                    delta_p_z = section.total_delta_P
                else:
                    delta_p_z = section.delta_p_z
                child1 = self.get_section(section.ch1)
                child2 = self.get_section(section.ch2)
                section.max_delta_P = max(child1.max_delta_P,
                                          child2.max_delta_P)  \
                    + delta_p_z

    def update_opt_fan_press(self):

        z_1 = (self.total_airflow*self.energy_cost*self.oper_time*self.PWEF
               )/(self.efficiency_fan_op*self.efficiency_motor*10**5)
        z_2 = (0.959 * np.pi * (self.density/self.dim_const)**0.2
               * self.duct_cost)

        self.opt_fan_press = (0.26 * (z_2*self.root.Kt / z_1)**(5/6)
                              + self.root.max_delta_P)

    def system_length(self):
        return len(self.sections)

    def data_to_csv(self, fName):

        with open("{}.csv".format(fName), "w") as f:
            for (i, s) in self.sections.items():
                section_dict = s.__dict__
                w = csv.DictWriter(f, section_dict.keys())
                if i == 1:
                    w.writeheader()
                w.writerow(section_dict)

    def export_system(self, fName):
        self.systemDF.to_csv(fName)
