import matplotlib.pyplot as plt
import numpy as np


def f1(x):
    return 4/(1+3*np.exp(-4*0.2*x))


def f2(x):
    return x**2

# evenly sampled time at 200ms intervals

xmin, xmax = -5, 5
ymin, ymax = -5, 5
size = 13

x1 = np.arange(0, 5, 0.01)
y1 = np.ones(np.shape(x1))*4
x7 = np.arange(0, np.log(3)/(4*0.2), 0.1)
y7 = np.ones(np.shape(x7))*f1(np.log(3)/(4*0.2))
y2 = np.arange(0, f1(np.log(3)/(4*0.2)), 0.01)
x2 = np.ones(np.shape(y2))*np.log(3)/(4*0.2)
y8 = np.arange(0, f1(2.3), 0.01)
x8 = np.ones(np.shape(y8))*2.3
y6 = np.arange(0, 3.25, 0.01)
x6 = np.ones(np.shape(y6))*4
x3 = np.arange(0., 5, 0.001)
y_i = np.array([f1(1) for i in range(500)]+[f1(1.5) for i in range(500)]+[f1(2) for i in range(500)]+
               [f1(2.5) for i in range(500)]+[f1(3) for i in range(500)]+[f1(3.5) for i in range(500)])

y_i2 = np.arange(0, f1(1.5), 0.1)
x_i2 = np.ones(np.shape(y_i2))*1.5
y_i3 = np.arange(0, f1(2), 0.1)
x_i3 = np.ones(np.shape(y_i3))*2
y_i4 = np.arange(0, f1(2.5), 0.1)
x_i4 = np.ones(np.shape(y_i4))*2.5
y_i5 = np.arange(0, f1(3), 0.1)
x_i5 = np.ones(np.shape(y_i5))*3
y_i6 = np.arange(0, f1(3.5), 0.1)
x_i6 = np.ones(np.shape(y_i6))*3.5
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
#ax.annotate('y=1', xy=(0., 0.), xytext=(3, 1.3), size=size)
#ax.annotate('-5', xy=(0., 0.), xytext=(-6.9, -0.9), size=size)
#ax.annotate(u"\u221A"+"300", xy=(0., 0.), xytext=(14.5, -3.5), size=size)
#ax.annotate('1', xy=(0., 0.), xytext=(-0.35, 1.1), size=size)
#ax.annotate('a', xy=(0., 0.), xytext=(0.9, -0.45), size=size)
#ax.annotate('b', xy=(0., 0.), xytext=(3.9, -0.45), size=size)
#ax.annotate('x', xy=(0., 0.), xytext=(2.9, -0.45), size=size)
#ax.annotate('A(x)', xy=(0., 0.), xytext=(1.9, 2.5), size=size)

#ax.fill_between(x1, f1(x1), facecolor="none", hatch='\\', lw=0.01)
#ax.fill_between(np.arange(3, 3.5, 0.01), f1(np.arange(3, 3.5, 0.01)), facecolor="none", hatch='\\')

axes = plt.gca()
axes.set_xlim([xmin, xmax])
axes.set_ylim([ymin, ymax])



# removing the default axis on all sides:
for side in ['bottom', 'right', 'top', 'left']:
    ax.spines[side].set_visible(False)


# removing the axis ticks
plt.xticks([])  # labels
plt.yticks([])
#ax.text(-0.2, ymax-0.3, r'$y$', ha='center', size=size)
#ax.text(xmax-0.2, -0.3, r'$x$', va='center', size=size)

ax.xaxis.set_ticks_position('none')  # tick markers
ax.yaxis.set_ticks_position('none')
dps = fig.dpi_scale_trans.inverted()
bbox = ax.get_window_extent().transformed(dps)
width, height = bbox.width, bbox.height

# manual arrowhead width and length
hw = 1./50.*(ymax-ymin)
hl = 1./50.*(xmax-xmin)
lw = 1. # axis line width
ohg = 0.3 # arrow overhang

# compute matching arrowhead length and width
yhw = hw/(ymax-ymin)*(xmax-xmin)* height/width
yhl = hl/(xmax-xmin)*(ymax-ymin)* width/height
'''
ax.arrow(2.3, 2.5, -0.3, f1(2)-2.5 , fc='k', ec='k', lw = lw,
         head_width=hw, head_length=hl, overhang = ohg,
         length_includes_head= True, clip_on = False)

ax.arrow(18, 20, -3, f(15)-20, fc='k', ec='k', lw = lw,
         head_width=hw, head_length=hl, overhang = ohg,
         length_includes_head= True, clip_on = False)
ax.arrow(4, -1, 0, 1, fc='k', ec='k', lw = lw,
         head_width=hw, head_length=hl, overhang = ohg,
         length_includes_head= True, clip_on = False)
ax.arrow(-2, 4, 1, 0, fc='k', ec='k', lw = lw,
         head_width=hw, head_length=hl, overhang = ohg,
         length_includes_head= True, clip_on = False)
'''

# draw x and y axis
ax.arrow(xmin, 0, xmax-xmin, 0., fc='k', ec='k', lw = lw,
         head_width=hw, head_length=hl, overhang = ohg,
         length_includes_head= True, clip_on = False)

ax.arrow(0, ymin, 0., ymax-ymin, fc='k', ec='k', lw = lw,
         head_width=yhw, head_length=yhl, overhang = ohg,
         length_includes_head= True, clip_on = False)

plt.plot(x3, f1(x3), 'k-', x7, y7, 'k--', x1, y1, 'k--', x2, y2, 'k--')
        # 0.4*np.cos(np.arange(0, 2*np.pi, 0.01))-3, 0.4*np.sin(np.arange(0, 2*np.pi, 0.01))+2, 'k-',
         #np.ones((17))*(-3), np.arange(0, 1.7, 0.1), 'k--', np.arange(-2.6, 0, 0.1), np.ones(26)*2, 'k--')
plt.show()