import numpy as np
import cv2
from tkinter import ttk
import tkinter as tk
from tkinter import filedialog as filedialog
import pandas as pd
import matplotlib.pyplot as plt
from ttkthemes import ThemedTk
from scipy.interpolate import interp1d
import os

#### VARIABLES ####
path = None
points = []
polynome_order = 3
area_new = pd.DataFrame()
y_max = None
x_max = None
y_min = None
x_min = None
x_min_picked = None
x_max_picked = None
y_min_picked = None
y_max_picked = None
plt.style.use('default')
counter_lines = 0
global_vars = globals()
#### RESIZING IMAGE FOR COMFORTABLE VIEW (DEPENDS ON MONITOR RESOLUTION) ####
def image_resize(image, width=None, height=None, inter=cv2.INTER_AREA):
    dim = None
    (h, w) = image.shape[:2]
    if width is None and height is None:
        return image
    if width is None:
        r = height / float(h)
        dim = (int(w * r), height)
    else:
        r = width / float(w)
        dim = (width, int(h * r))
    resized = cv2.resize(image, dim, interpolation=inter)
    return resized


#### OPEN IMAGE ####
def open_image():

    global path, f, f_new, img_height, img_width, color_copy, color_copy_crop, path

    def nothing(x):
        pass
    
    path = filedialog.askopenfilename()
    if path == '':
        return
    f = cv2.imread(path)
    f = image_resize(f, height=800, width=800)
    f_new = f.copy()
    color_copy = f_new.copy()
    color_copy = cv2.resize(color_copy, (500, 500),
                            interpolation=cv2.INTER_LANCZOS4)
    color_copy_crop = f_new.copy()
    cv2.imshow("image", f)
    img_height, img_width, channels = f.shape


    restart_button.config(state=tk.NORMAL)
    Entry_xmin.config(state=tk.NORMAL)
    Entry_ymin.config(state=tk.NORMAL)
    Entry_xmax.config(state=tk.NORMAL)
    Entry_ymax.config(state=tk.NORMAL)
    pick_x_min_button.config(state=tk.NORMAL)
    pick_y_min_button.config(state=tk.NORMAL)
    pick_x_max_button.config(state=tk.NORMAL)
    pick_y_max_button.config(state=tk.NORMAL)
    hsv_colormap_button.configure(state=tk.NORMAL)
    pick_data_points_button.configure(state=tk.NORMAL)
    done_button.configure(state=tk.NORMAL)
    clear_img_btn.configure(state=tk.NORMAL)

#### RESTART PROGRAM ####
def restart():

    global points, num_elem, x_min, x_max, y_min, y_max, area_new, counter_lines
    counter_lines = 0
    area_new = pd.DataFrame()
    pick_x_min_button.config(state=tk.DISABLED)
    pick_y_min_button.config(state=tk.DISABLED)
    pick_x_max_button.config(state=tk.DISABLED)
    pick_y_max_button.config(state=tk.DISABLED)
    hsv_colormap_button.configure(state=tk.DISABLED)
    pick_data_points_button.configure(state=tk.DISABLED)
    curve_fitting_button.configure(state=tk.DISABLED)
    interpolation_fitting_button.config(state=tk.DISABLED)
    picked_points_button.config(state=tk.DISABLED)
    area_btn.config(state=tk.DISABLED)
    choose_another_line_button.config(state=tk.DISABLED)
    combobox.configure(state=tk.DISABLED)
    Entry_xmin.config(state=tk.NORMAL)
    Entry_xmin.delete(0, tk.END)
    Entry_xmin.config(state=tk.DISABLED)
    Entry_ymin.config(state=tk.NORMAL)
    Entry_ymin.delete(0, tk.END)
    Entry_ymin.config(state=tk.DISABLED)
    Entry_xmax.config(state=tk.NORMAL)
    Entry_xmax.delete(0, tk.END)
    Entry_xmax.config(state=tk.DISABLED)
    Entry_ymax.config(state=tk.NORMAL)
    Entry_ymax.delete(0, tk.END)
    Entry_ymax.config(state=tk.DISABLED)
    clear_img_btn.configure(state=tk.DISABLED)
    name_file.set('')
    Data_get_name.config(state=tk.DISABLED)
    points = []  # points
    radio_interpolate.config(state=tk.DISABLED)
    radio_picked_points.config(state=tk.DISABLED)
    radio_curv_fit.config(state=tk.DISABLED)
    done_button.config(state=tk.DISABLED)
    num_elem_ent.set('5000')
    # INITIAL CONDITIONS OF AXIS LIMITS
    y_max = None
    x_max = None
    y_min = None
    x_min = None
    num_elem = 0
    num_of_elem_approx.config(state=tk.DISABLED)
    
    try:
        cv2.destroyWindow("image")
    except:
        pass

#### CLEAR IMAGE ####
def clear_image():
    global f, color_copy, color_copy_crop, points,y_max, x_max, y_min, x_min, x_min_picked, x_max_picked, y_min_picked, y_max_picked
    
    curve_fitting_button.config(state=tk.DISABLED)
    interpolation_fitting_button.config(state=tk.DISABLED)
    picked_points_button.config(state=tk.DISABLED)
    radio_interpolate.config(state=tk.DISABLED)
    radio_picked_points.config(state=tk.DISABLED)
    radio_curv_fit.config(state=tk.DISABLED)
    area_btn.config(state=tk.DISABLED)
    choose_another_line_button.config(state=tk.DISABLED)
    hsv_colormap_button.config(state=tk.NORMAL)
    combobox.config(state=tk.DISABLED)
    num_of_elem_approx.config(state=tk.DISABLED)
    reply = tk.messagebox.askyesno('QuestionBox', 'Do you want to save axis limits?')
        
    f = f_new.copy()
    color_copy = f_new.copy()
    color_copy = cv2.resize(color_copy, (500, 500),
                            interpolation=cv2.INTER_LANCZOS4)
    color_copy_crop = f_new.copy()  # clear image
    points = []  # clear data of points
    cv2.imshow("image", f)
    
    if reply:
        if x_min_picked:
            cv2.circle(f,x_min_picked,5, (255,255,0), -1)
            cv2.putText(f, "x_min", (x_min_picked), 2, 1, (255, 255, 0))
        if x_max_picked:
            cv2.circle(f,x_max_picked,5, (255,255,0), -1)
            cv2.putText(f, "x_max", (x_max_picked), 2, 1, (255, 255, 0))
        if y_min_picked:
            cv2.circle(f,y_min_picked,5, (255,255,0), -1)
            cv2.putText(f, "y_min", (y_min_picked), 2, 1, (255, 255, 0))
        if y_max_picked:
            cv2.circle(f,y_max_picked,5, (255,255,0), -1)
            cv2.putText(f, "y_max", (y_max_picked), 2, 1, (255, 255, 0))

        cv2.imshow('image', f)
    
    
    else:
        
        y_max = None
        x_max = None
        y_min = None
        x_min = None
    

def points_pick(event, x, y, flags, param):
    
    global points, x1, y1

    if event == cv2.EVENT_LBUTTONDOWN:
        if points:
            cv2.line(f, (x1, y1), (x, y), (255, 255, 0), 2)
            cv2.imshow("image", f)
        # ADDING POINTS
        points.append([x, y])
        x1 = x
        y1 = y
        
def axis(event, x, y, flags, param):
    global f
    if event == cv2.EVENT_MOUSEMOVE:
        f_copy = f.copy()
        cv2.line(f, (0, y), (img_width, y), (255, 255, 0), 2)
        cv2.line(f, (x, 0), (x, img_height), (255, 255, 0), 2)
        cv2.imshow('image', f)
        f = f_copy.copy()
    elif event == cv2.EVENT_LBUTTONDOWN:
        cv2.circle(f, (x, y), 5, (255, 255, 0), -1)
        cv2.putText(f, f"{param}", (x, y), 2, 1, (255, 255, 0))
        global_vars[f"{param}_picked"] = (x, y)
        cv2.imshow("image", f)
        global_vars[f"{param}"] = x if param[0] == 'x' else y
        cv2.setMouseCallback('image', lambda *args: None)

def pick_axis(param):
    tk.messagebox.showinfo('INFO', f'Pick {param}')
    cv2.setMouseCallback("image", axis, param)
    
def pick_points():
    tk.messagebox.showinfo('INFO', 'Pick Data Points')
    cv2.setMouseCallback("image", points_pick)



#### TRANSFER COORDINATES OF POINTS FROM NUMPY TO MATPLOTLIB ####
def done():
    
    global num_elem, db, database_curve_fit, data_interpolate, x_curve_fit, y_curve_fit, x_interpolate_fit, y_interpolate_fit, k_035, k_035_curve

    try:

        num_elem = int(num_elem_ent.get())
        num_elem_ent.set(num_elem)

        if num_elem < 0:
            raise tk.messagebox.showerror(
                "ERROR", "Number of points must be positive")
            
    except ValueError:
        
        raise tk.messagebox.showerror(
            "ERROR", "Incorrect type of input(Number of elements), number of elements must be integer and positive ")

    try:
        
        ux_min = float(entry_xmin_var.get().replace(',', '.'))
        uy_min = float(entry_ymin_var.get().replace(',', '.'))
        ux_max = float(entry_xmax_var.get().replace(',', '.'))
        uy_max = float(entry_ymax_var.get().replace(',', '.'))
        
    except ValueError:
        
        return tk.messagebox.showerror("ERROR", "Incorrect type of input(xmin, xmax, ymin, ymax), format of numbers must be: 1 or 1,0 or 1.0 ")


    try:
        
        a = (float(ux_max)-float(ux_min))/(x_max-x_min)
        b = (float(uy_max)-float(uy_min))/(y_max-y_min)
        
    except TypeError:
        
        return tk.messagebox.showerror("ERROR", "Limits of axis not Found")
    points_to_float = np.asfarray(points)

    try:
        
        points_to_float [:, 0] = float(ux_min) + a*(points_to_float [:, 0] - float(x_min))
        points_to_float [:, 1] = float(uy_min) + b*(points_to_float [:, 1] - float(y_min))
        
    except IndexError:
        
        return tk.messagebox.showerror("ERROR", "Data Points not Found")

    db = pd.DataFrame(points_to_float , columns=['x', 'y'])
    # SORTING DATAFRAME
    x = np.array(db.x)
    y = np.array(db.y)
    db.sort_values('x')

    # INTERPOLATE FITTING
    func = interp1d(x, y, fill_value='extrapolate')
    x_interpolate_fit = np.linspace(float(x[0]), float(
        x[len(x)-1]), num=num_elem, endpoint=True)
    y_interpolate_fit = func(x_interpolate_fit)
    data_interpolate = pd.DataFrame(columns=['x_interpolate', 'y_interpolate'])
    data_interpolate.x_interpolate = x_interpolate_fit
    data_interpolate.y_interpolate = y_interpolate_fit

    # CURVE FITTING
    coef_1 = np.polyfit(x, y, polynome_order)  # polynomial fitting
    pol_1 = np.poly1d(coef_1)
    x_curve_fit = np.linspace(float(db.x[0]), float(
        db.x[len(x)-1]), num=num_elem, endpoint=True)
    y_curve_fit = pol_1(x_curve_fit)
    database_curve_fit = pd.DataFrame(columns=['x_curve_fit', 'y_curve_fit'])
    database_curve_fit.x_curve_fit = x_curve_fit
    database_curve_fit.y_curve_fit = y_curve_fit



    curve_fitting_button.configure(state=tk.NORMAL)
    interpolation_fitting_button.config(state=tk.NORMAL)
    picked_points_button.config(state=tk.NORMAL)
    combobox.configure(state='readonly')
    radio_interpolate.config(state=tk.NORMAL)
    radio_picked_points.config(state=tk.NORMAL)
    radio_curv_fit.config(state=tk.NORMAL)
    num_of_elem_approx.config(state=tk.NORMAL)
    area_btn.config(state=tk.NORMAL)
    Data_get_name.config(state=tk.NORMAL)
    choose_another_line_button.config(state=tk.NORMAL)



#### CHECKBOX FITTING POLYNOMIAL ####
def changed_func(event):

    global polynome_order
    combobox.selection_clear()
    polynome_order = int(combobox.get()[0])
    done()

#### POLYNOMIAL FITTING ####
def fitting():
    
    done()
    fig1, ax1 = plt.subplots()
    ax1.set_title('Curve Fitting \n{} points\n{}'.format(num_elem, combobox.get()))
    ax1.plot(db.x, db.y, 'ro', x_curve_fit, y_curve_fit, '-')
    ax1.grid()
    plt.show(block=False)


def picked_points_plot():
    
    done()
    fig2, ax2 = plt.subplots()
    ax2.set_title('Picked points')
    ax2.plot(db.x, db.y, 'o')
    ax2.grid()
    plt.show(block=False)


#### INTERPOLATION FIT ####
def interpolation_fit():
    
    global f
    done()
    fig3, ax3 = plt.subplots()
    ax3.plot(db.x, db.y, 'ro', x_interpolate_fit, y_interpolate_fit, '-')
    ax3.set_title('Interpolation \n%s points' % num_elem)
    ax3.grid()
    plt.show(block=False) 
    
    
#### DETECTION COLOR ON THE IMAGE ####
def autodetect_color():
    
    global f, points, white_img
    cv2.destroyWindow("HSV Colormap")
    copy_image = white_img.copy()
    second_copy = white_img.copy()
    second_copy = cv2.cvtColor(white_img, cv2.COLOR_BGR2HSV)  # RGB -> HSV
    mask = cv2.inRange(second_copy, hsv_min1, hsv_max1)  # mask
    copy_image = cv2.bitwise_and(
        copy_image, copy_image, mask=mask)  # apply mask
    points = cv2.findNonZero(mask)  # coordinates of mask
    n = np.shape(points)[0]
    points.shape = (n, 2)

    for i in range(0, n-1):
        cv2.circle(f, ((points[i][0]), (points[i][1])), 3, (255, 255, 0), -1)
        cv2.imshow("image", f)

#### CROP IMAGE ####
def crop_image_click():
    tk.messagebox.showinfo(
        title='Information', message='Select a rectangle(drag Left Button Mouse) which you want to convert to HSV')
    cv2.setMouseCallback('image', crop_image)  # catch mouse event


def crop_image(event, x, y, flags, param):

    global white_img, drawing, cropping, f, point1, point2, min_x, min_y, width, height_rect, color_copy, f_new, color_copy_crop, width_rect
    
    f_third_copy = f.copy()

    if event == cv2.EVENT_LBUTTONDOWN:

        drawing = True
        point1 = (x, y)
        cropping = True

    elif event == cv2.EVENT_MOUSEMOVE:

        try:
            if drawing == True:
                cv2.rectangle(f, point1, (x, y), (0, 255, 0), 2)
                cv2.imshow('image', f)
                f = f_third_copy.copy()
        except NameError:
            pass
    
    elif event == cv2.EVENT_LBUTTONUP:
        point2 = (x, y)
        drawing = False
        min_x = min(point1[0], point2[0])
        min_y = min(point1[1], point2[1])
        width_rect = abs(point1[0] - point2[0])
        height_rect = abs(point1[1] - point2[1])
        cropping = False
        white_img = np.ones((f.shape), np.uint8) * 255
        color_copy_crop = color_copy_crop[min_y:min_y +
                                          height_rect, min_x:min_x+width_rect]
        white_img[min_y:min_y+height_rect, min_x:min_x +
                  width_rect] = color_copy_crop.copy()
        color_copy = white_img.copy()
        f = f_third_copy.copy()
        cv2.imshow('image', f)
        hsv_colorspace_detect()


#### DETECT COLOR CURVE BY HSV MASK ####
def hsv_colorspace_detect():
    global hsv_min1, hsv_max1, f, detect_color_window, color_copy

    tk.messagebox.showinfo(title='Information', message='Press q when finish')

    def nothing(x):
        pass

    image = color_copy_crop.copy()  # cropped image

    cv2.namedWindow('HSV Colormap', cv2.WINDOW_AUTOSIZE)  # create cv2 window

    # CREATE TRACKBARS
    cv2.createTrackbar('HMin', 'HSV Colormap', 0, 179, nothing)
    cv2.createTrackbar('SMin', 'HSV Colormap', 0, 255, nothing)
    cv2.createTrackbar('VMin', 'HSV Colormap', 0, 255, nothing)
    cv2.createTrackbar('HMax', 'HSV Colormap', 0, 179, nothing)
    cv2.createTrackbar('SMax', 'HSV Colormap', 0, 255, nothing)
    cv2.createTrackbar('VMax', 'HSV Colormap', 0, 255, nothing)

    cv2.setTrackbarPos('HMax', 'HSV Colormap', 179)
    cv2.setTrackbarPos('SMax', 'HSV Colormap', 255)
    cv2.setTrackbarPos('VMax', 'HSV Colormap', 255)

    hMin = sMin = vMin = hMax = sMax = vMax = 0

    while(1):
        # GET CURRENT POS ON TRACKBARS
        hMin = cv2.getTrackbarPos('HMin', 'HSV Colormap')
        sMin = cv2.getTrackbarPos('SMin', 'HSV Colormap')
        vMin = cv2.getTrackbarPos('VMin', 'HSV Colormap')
        hMax = cv2.getTrackbarPos('HMax', 'HSV Colormap')
        sMax = cv2.getTrackbarPos('SMax', 'HSV Colormap')
        vMax = cv2.getTrackbarPos('VMax', 'HSV Colormap')

        lower = np.array([hMin, sMin, vMin])  # lower value hsv mask
        upper = np.array([hMax, sMax, vMax])  # upper mask hsv

        # CONVERT TO HSV FORMAT
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)  # RGB -> HSV
        # geting all colors between lower and upper values of hsv (result will be shown on image)
        mask = cv2.inRange(hsv, lower, upper)
        # get a subset of an image defined by another image, typically referred to as a "mask"
        result = cv2.bitwise_and(image, image, mask=mask)

        # DISPLAY RESULT IMAGE
        cv2.imshow('HSV Colormap', result)
        if cv2.waitKey(10) & 0xFF == ord('q'):
            hsv_min1 = (hMin, sMin, vMin)
            hsv_max1 = (hMax, sMax, vMax)
            autodetect_color()
            break
        



#### CREATING CSV DATA ####
def another_line(with_clear=True):
    global area_new, counter_lines
    
    if with_clear:
        clear_image()

    if radiobtn.get() == 1:
        area_new['x%d' % counter_lines] = data_interpolate.x_interpolate
        area_new['y%d' % counter_lines] = data_interpolate.y_interpolate
    elif radiobtn.get() == 2:
        area_new['x%d' % counter_lines] = database_curve_fit.x_curve_fit
        area_new['y%d' % counter_lines] = database_curve_fit.y_curve_fit
    elif radiobtn.get() == 3:
        area_new['x%d' % counter_lines] = db.x
        area_new['y%d' % counter_lines] = db.y
    counter_lines += 1




def save_data():
    
    
    another_line(with_clear = False)
    file = name_file.get()
    area_new.to_csv('./%s.csv' % file, mode='w', index=False)
    tk.messagebox.showinfo(title='Successful', message='%s.csv Saved!' % file)
    database.config(
        values=[fname for fname in os.listdir(folder) if fname[-3:] == 'csv'])
    restart()

def changed_csv(event):
    global abs_data, abs_name, yieldstrap_data
    database.selection_clear()
    csv_name = database.get()
    abs_data = pd.read_csv('./%s' % csv_name, delimiter=',')
    for i in range(int(list(abs_data.columns)[-1][-1])+1):
        plt.plot(abs_data[f'x{i}'], abs_data[f'y{i}'])
    plt.grid()
    plt.show(block=False)


#INITIALIZE THE WINDOW TOOLKIT ####
# library with autodesign widgets in tkinter, more themes here -> https://ttkthemes.readthedocs.io/
root = ThemedTk(theme="adapta")
root.title("BeDigitizer")
root.geometry("576x530")
root.resizable(False, False)

main_frame = ttk.Frame(root, width=576, height=530)
main_frame.place(x=0, y=0)



select_image_button = ttk.Button(
    main_frame, text="Select an image", command=open_image, width=15)
select_image_button.place(x=0, y=0)

restart_button = ttk.Button(
    main_frame, text="Restart", command=restart, width=15)
restart_button.place(x=115, y=0)

clear_img_btn = ttk.Button(main_frame, text="Clear Image",
                           command=clear_image, width=15, state=tk.DISABLED)
clear_img_btn.place(x=230, y=0)




entry_xmin_var = tk.StringVar()
Entry_xmin = ttk.Entry(main_frame, width=12,
                       state=tk.DISABLED, textvariable=entry_xmin_var)
Entry_xmin.place(x=115, y=58)

entry_ymin_var = tk.StringVar()
Entry_ymin = ttk.Entry(main_frame, width=12,
                       state=tk.DISABLED, textvariable=entry_ymin_var)
Entry_ymin.place(x=115, y=98)

entry_xmax_var = tk.StringVar()
Entry_xmax = ttk.Entry(main_frame, width=12,
                       state=tk.DISABLED, textvariable=entry_xmax_var)
Entry_xmax.place(x=115, y=138)

entry_ymax_var = tk.StringVar()
Entry_ymax = ttk.Entry(main_frame, width=12,
                       state=tk.DISABLED, textvariable=entry_ymax_var)
Entry_ymax.place(x=115, y=178)

pick_x_min_button = ttk.Button(
    main_frame, text="Pick x min", command=lambda: pick_axis('x_min'), width=15, state=tk.DISABLED)
pick_x_min_button.place(x=0, y=50)

pick_y_min_button = ttk.Button(
    main_frame, text="Pick y min", command=lambda: pick_axis('y_min'), width=15, state=tk.DISABLED)
pick_y_min_button.place(x=0, y=90)

pick_x_max_button = ttk.Button(
    main_frame, text="Pick x max", command=lambda: pick_axis('x_max'), width=15, state=tk.DISABLED)
pick_x_max_button.place(x=0, y=130)

pick_y_max_button = ttk.Button(
    main_frame, text="Pick y max", command=lambda: pick_axis('y_max'), width=15, state=tk.DISABLED)
pick_y_max_button.place(x=0, y=170)




pick_data_points_button = ttk.Button(
    main_frame, text="Pick data points", command=pick_points, width=15, state=tk.DISABLED)
pick_data_points_button.place(x=270, y=70)

hsv_colormap_button = ttk.Button(
    main_frame, text="HSV Colormap", command=crop_image_click, width=15, state=tk.DISABLED)
hsv_colormap_button.place(x=270, y=150)




done_button = ttk.Button(main_frame, text="Done",
                         command=done, state=tk.DISABLED, width=15)
done_button.place(x=460, y=110)




curve_fitting_button = ttk.Button(
    main_frame, text="Polynomial Fitting", command=fitting, width=25, state=tk.DISABLED)
curve_fitting_button.place(x=60, y=300)

interpolation_fitting_button = ttk.Button(
    main_frame, text="Interpolation", command=interpolation_fit, width=25, state=tk.DISABLED)
interpolation_fitting_button.place(x=60, y=260)

picked_points_button = ttk.Button(
    main_frame, text="Picked points", command=picked_points_plot, width = 25, state=tk.DISABLED)
picked_points_button.place(x=60, y=220)

num_elem_ent = tk.IntVar()
num_of_elem_approx = ttk.Entry(
    main_frame, width=12, state=tk.DISABLED, textvariable=num_elem_ent)
num_of_elem_approx.place(x=420, y=260)
num_elem_ent.set('5000')





choose_another_line_button = ttk.Button(
    main_frame, text="Choose another line", command=another_line, width=71, state=tk.DISABLED)
choose_another_line_button.place(x=60, y=350)

area_btn = ttk.Button(main_frame, text="SAVE",  width=15,
                      command=save_data, state=tk.DISABLED)
area_btn.place(x=380, y=420)

ttk.Label(main_frame, text="Enter a name of data file",
          font=('slant', 9)).place(x=80, y=410)

name_file = tk.StringVar()
Data_get_name = ttk.Entry(
    main_frame, textvariable=name_file, width=45, state=tk.DISABLED)
Data_get_name.place(x=80, y=426)





folder = os.path.realpath('./')
filelist = [fname for fname in os.listdir(folder) if fname[-3:] == 'csv']
database = ttk.Combobox(main_frame, state='readonly',
                        values=filelist, text="Choose csv", width=64)
database.place(x=80, y=460)
database.set('Choose a csv to see a plot')
database.bind('<<ComboboxSelected>>', changed_csv)
database.selection_clear()




radiobtn = tk.IntVar()
radiobtn.set(3)
radio_interpolate = ttk.Radiobutton(
    main_frame, variable=radiobtn, value=1, state=tk.DISABLED)
radio_interpolate.place(x=30, y=264)
radio_curv_fit = ttk.Radiobutton(
    main_frame, variable=radiobtn, value=2, state=tk.DISABLED)
radio_curv_fit.place(x=30, y=304)
radio_picked_points = ttk.Radiobutton(main_frame, variable=radiobtn, value = 3, state=tk.DISABLED)
radio_picked_points.place(x=30, y=224)





ttk.Label(main_frame, text="=>").place(x=230, y=120)
ttk.Label(main_frame, text="=>").place(x=400, y=120)
ttk.Label(main_frame, text="OR").place(x=320, y=120)
ttk.Label(main_frame, text="Enter number of points for fitting:",
          font=('slant', 9)).place(x=240, y=265)

cmbvar = tk.StringVar()
combobox = ttk.Combobox(main_frame, textvariable=cmbvar, width=40, state=tk.DISABLED, values=["3th degree polynomial",
                                                                                              "4th degree polynomial",
                                                                                              "5th degree polynomial",
                                                                                              "6th degree polynomial",
                                                                                              "7th degree polynomial"])
combobox.place(x=240, y=295)
combobox.current(0)
combobox.bind('<<ComboboxSelected>>', changed_func)





root.mainloop()