********************
Using TARDIS Widgets
********************

This page describes what each TARDIS Widget has to offer and how you can make
the best use of it. If you're looking for the code to generate widgets, head
over to `Generating Custom Abundance Widget <how_to_abundance_widget>`_ section or 
`Generating Data Exploration Widgets <how_to_generating_widgets>`_ section to see the
notebook in action.

Currently, TARDIS supports the following widgets:

Custom Abundance Widget
#######################
This widget (or GUI) allows you to develop custom model compositions 
graphically and output the model to a new file.

.. image:: ../images/custom_abundance_widget.png
    :alt: Demo of Custom Abundance Widget

The GUI consists of three parts:

1. **Visualization plot** - An interactive step graph that shows abundances 
and densities as a function of velocity. 

2. **Data editor** - An interface to edit abundances, densities and velocity 
shells.

3. **File output** - An output module to save the model compositions as a CSVY 
file.

Interacting with the GUI
========================

You can interact with this GUI in the following ways:

Edit Abundances
---------------
There are two radio buttons which allow you to edit either single shell or 
multiple shells. The new input will be applied to selected shell(s) immediately 
and the plot is updated at the same time. If you want to edit multiple shells 
at a time, remember to choose the second radio button and set the range of 
shell number using int slider. The selected shell(s) is highlighted in the 
plot.

.. image:: ../images/cus_abund_edit_abundance.gif
    :alt: Demo of editing abundances

Normalize Abundances
--------------------
Click `Normalize` button to normalize the abundances on selected shell(s) to 
1. If you wish to keep the abundance of a certain element unchanged during the 
normalization, you can select the checkbox near that element to lock it. 
The normalization will be applied to other elements and the sum of the 
abundances still remains at 1.

.. image:: ../images/cus_abund_norm.gif
    :alt: Demo of normalization

Edit Densities
--------------
In `Edit Density` tab, you also can edit either single shell or the whole 
density profile. To calculate a new profile, you need to choose the density 
type and input required parameters.

.. image:: ../images/cus_abund_edit_density.gif
    :alt: Demo of editing densities

Add New Element
---------------
At the bottom of the editor, the symbol input box allows you to add new element 
or isotope to the model. It automatically recognizes whether the symbol exists 
or whether the element is already in the model.

.. image:: ../images/cus_abund_add_element.gif
    :alt: Demo of adding new element

Add New Shell
-------------
Another important functionality is to add new shell to the model. You need to 
specify the velocity range of the new shell and abundances at that new shell 
will be set to 0. Note that the existing shell(s) will be resized smartly if 
the new shell overlaps with it.

.. image:: ../images/cus_abund_add_shell.gif
    :alt: Demo of adding new shell

Shell Info Widget
#################

This widget allows you to explore the chemical abundances in different shells
of the model Supernova ejecta.

.. image:: ../images/shell_info_widget_demo.gif
    :alt: Demo of Shell Info Widget

It consists of four interlinked tables - clicking on any row in a table,
populates data in the table(s) to the right of it. Each table has the
following information:

1. **Shells Data** - Radiative temperature and Dilution Factor (W) of each shell
(computational partitions) of the model Supernova ejecta. Shell numbers are
given in ascending order, from the innermost shell to the outermost.

2. **Element Abundances** - Fractional mass abundance of each element present
in the selected shell.

3. **Ion Abundances** - Fractional mass abundance of each ion (species) of the
selected element present in the selected shell. 

4. **Level Abundances** - Fractional mass abundance of each level of the
selected ion and element in the selected shell.

Try the interactive demo below (click the run button to launch it in your
browser using Pyodide — no installation required):

.. pyodide::

    import pandas as pd
    import panel as pn

    pn.extension("tabulator")

    # Sample data representing a TARDIS simulation output
    def roman(n):
        vals = [
            (1000, "M"), (900, "CM"), (500, "D"), (400, "CD"),
            (100, "C"), (90, "XC"), (50, "L"), (40, "XL"),
            (10, "X"), (9, "IX"), (5, "V"), (4, "IV"), (1, "I"),
        ]
        result = ""
        for v, s in vals:
            while n >= v:
                result += s
                n -= v
        return result

    def fmt(x):
        return f"{x:.6e}"

    n_shells = 5
    t_rad = [10000, 9500, 9000, 8500, 8000]
    dilution = [0.10, 0.15, 0.20, 0.25, 0.30]
    shells_df = pd.DataFrame(
        {
            "Rad. Temp.": [fmt(t) for t in t_rad],
            "Dilution Factor": [fmt(w) for w in dilution],
        },
        index=pd.Index(range(1, n_shells + 1), name="Shell No."),
    )

    element_symbols = {1: "H", 14: "Si", 26: "Fe"}
    element_Z = [1, 14, 26]
    element_ab = {
        1:  [0.50, 0.50, 0.50, 0.50, 0.50],
        14: [0.30, 0.30, 0.30, 0.30, 0.30],
        26: [0.20, 0.20, 0.20, 0.20, 0.20],
    }
    ion_counts_map = {1: 2, 14: 2, 26: 3}
    ion_ab = {
        (1, 0):  [0.90, 0.85, 0.80, 0.75, 0.70],
        (1, 1):  [0.10, 0.15, 0.20, 0.25, 0.30],
        (14, 0): [0.60, 0.55, 0.50, 0.45, 0.40],
        (14, 1): [0.40, 0.45, 0.50, 0.55, 0.60],
        (26, 0): [0.50, 0.48, 0.46, 0.44, 0.42],
        (26, 1): [0.30, 0.32, 0.34, 0.36, 0.38],
        (26, 2): [0.20, 0.20, 0.20, 0.20, 0.20],
    }
    level_ab = {
        (1, 0):  [0.70, 0.20, 0.10],
        (1, 1):  [0.80, 0.20],
        (14, 0): [0.50, 0.30, 0.20],
        (14, 1): [0.60, 0.40],
        (26, 0): [0.40, 0.35, 0.25],
        (26, 1): [0.50, 0.30, 0.20],
        (26, 2): [0.60, 0.40],
    }

    def get_element_df(shell_num):
        col = f"Frac. Ab. (Shell {shell_num})"
        return pd.DataFrame(
            {
                "Element": [element_symbols[z] for z in element_Z],
                col: [fmt(element_ab[z][shell_num - 1]) for z in element_Z],
            },
            index=pd.Index(element_Z, name="Z"),
        )

    def get_ion_df(z, shell_num):
        n_ions = ion_counts_map[z]
        col = f"Frac. Ab. (Z={z})"
        return pd.DataFrame(
            {
                "Species": [f"{element_symbols[z]} {roman(i + 1)}" for i in range(n_ions)],
                col: [fmt(ion_ab[(z, i)][shell_num - 1]) for i in range(n_ions)],
            },
            index=pd.Index(range(n_ions), name="Ion"),
        )

    def get_level_df(z, ion):
        levels = level_ab[(z, ion)]
        col = f"Frac. Ab. (Ion={ion})"
        return pd.DataFrame(
            {col: [fmt(a) for a in levels]},
            index=pd.Index(range(len(levels)), name="Level"),
        )

    def make_table(df):
        return pn.widgets.Tabulator(
            df,
            selectable=True,
            show_index=True,
            sizing_mode="stretch_width",
            height=min(400, max(200, len(df) * 30 + 50)),
            disabled=True,
        )

    shells_table = make_table(shells_df)
    elem_table = make_table(get_element_df(1))
    ion_table = make_table(get_ion_df(element_Z[0], 1))
    level_table = make_table(get_level_df(element_Z[0], 0))

    def on_shell_select(event):
        if not event.new:
            return
        shell_num = event.new[0] + 1
        elem_table.value = get_element_df(shell_num)
        if elem_table.selection == [0]:
            elem_table.selection = []
        elem_table.selection = [0]

    def on_elem_select(event):
        if not event.new:
            return
        shell_sel = shells_table.selection
        if not shell_sel:
            return
        shell_num = shell_sel[0] + 1
        z = element_Z[event.new[0]]
        ion_table.value = get_ion_df(z, shell_num)
        if ion_table.selection == [0]:
            ion_table.selection = []
        ion_table.selection = [0]

    def on_ion_select(event):
        if not event.new:
            return
        shell_sel = shells_table.selection
        elem_sel = elem_table.selection
        if not shell_sel or not elem_sel:
            return
        z = element_Z[elem_sel[0]]
        ion = event.new[0]
        level_table.value = get_level_df(z, ion)

    shells_table.param.watch(on_shell_select, "selection")
    elem_table.param.watch(on_elem_select, "selection")
    ion_table.param.watch(on_ion_select, "selection")

    shells_table.selection = [0]

    info_text = pn.pane.HTML(
        "<b>Frac. Ab.</b> denotes <i>Fractional Abundances</i> (all values in "
        "each table sum to 1)<br><b>W</b> denotes <i>Dilution Factor</i> and "
        "<b>Rad. Temp.</b> is <i>Radiative Temperature (in K)</i>"
    )

    pn.Column(
        info_text,
        pn.Row(shells_table, elem_table, ion_table, level_table, sizing_mode="stretch_width"),
    ).servable()

Line Info Widget
################

This widget lets you explore the atomic lines responsible for producing
features in the simulated spectrum.

.. image:: ../images/line_info_widget_demo.gif
    :alt: Demo of Line Info Widget

By selecting a wavelength range on the spectrum plot, you can see the species
that produced the features within that range. This is determined by counting
the number of escaping packets that experienced their last interaction with
each species. Packets can be filtered by the wavelength at which they were
absorbed or emitted, using the toggle buttons.

You can then click on a species to see the packets count for each last line
interaction it experienced. Using the dropdown menu, these counts can be grouped
by excitation lines, de-excitation lines, or both.

Interacting with Spectrum
=========================

The spectrum in the Line Info Widget is an interactive figure made using
`plotly <https://plotly.com/python/>`_, there are several things you can
do with it:

Making Selection
----------------
The box selection is enabled by default, so you just need to click and drag on
the figure and a pink colored selection box will appear. By making a
selection across the wavelength range you're interested in, tables update
to show line information of only packets from that range.

.. image:: ../images/line_info_spectrum_selection.gif
    :alt: Demo of making selection

After making a selection, if you need to resize the selection box (say, make it
narrower), simply redraw a new selection box over the older one.

Using Rangesilder
-----------------
The rangeslider is a long bar below the figure that allows you to zoom in on a
particular wavelength range in the long spectrum.

.. image:: ../images/line_info_spectrum_rangeslider.gif
    :alt: Demo of using rangeslider

Either you can **slide** the zoomed range by clicking and dragging it or you 
can **resize** it by dragging the handles (vertical bars) at its edges.

Using other options in Modebar
------------------------------
If you take your mouse to the top right corner of the figure, you will see a
Modebar with multiple options. The default option when Line Info Widget first
displays is **Box Select** - the dotted square icon. You can click on other
options like **Zoom** (magnifying glass icon), to do a rectangular zoom which
may be helpful to focus on a feature in the spectrum. You can always revert
back to the initial state of the figure by clicking on **Reset Axes** option.

.. image:: ../images/line_info_spectrum_modebar.gif
    :alt: Demo of using modebar options

There are also several other options in the modebar which we have not explained
(because they are not very relevant) but you're free to explore them as long as
you remember to click back on the **Box Select** option for making selections on
spectrum.

Energy Level Diagram
####################

This widget lets you visualize the last line interactions

.. image:: ../images/energy_level_widget_demo.gif
    :alt: Demo of Energy Level Diagram

By selecting an ion on the widget, you can see its energy level diagram, which
also shows information about the last line interactions experienced by packets
in the simulation.

The y-axis of the plot represents energy while the horizontal lines show
discrete energy levels. The thickness of each line represents the level
population, with thicker lines representing a greater population than the thin lines.

Arrows represent the line interactions experienced by packets. Upwards arrows
show excitation from lower energy levels to higher levels and downward arrows
show de-excitation from higher energy levels to lower levels. The thickness of
each arrow represents the number of packets that underwent that interaction,
with thicker lines representing more packets than the thin lines.
The wavelength of the transition is given by the color.

Setting Other Options
---------------------
You can select the range on which to filter the wavelength using the slider.
You can also select the model shell by which to filter the last line interactions
and the level populations. If no shell is selected, then all the last line
interactions are plotted and the level populations are averaged across all shells
in the simulation. You can also set the maximum number of levels to show on the plot.

Lastly, you can also set the scale of the y-axis: Linear or Log.

.. image:: ../images/energy_level_widget_options.gif
    :alt: Demo of using options

.. Toggle legend
