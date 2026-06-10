import numpy as np
import matplotlib.pyplot as plt
from acoular.tools import c_air

plt.figure(figsize=(9, 4))

plt.subplot(121)
for celsius in [0, 10, 20, 30]:
    c = []
    co2_values = np.linspace(0, 1., 100)
    for co2 in co2_values:
        c.append(c_air(celsius, 0.0, 101325, co2))
    plt.plot(co2_values, c, label=f'{celsius}°C', color='black')
    plt.ylabel('Speed of sound (m/s)')
    plt.xlabel('CO2 concentration (%)')
    plt.xlim(0, co2_values.max())
    plt.ylim(330, 355)
    plt.minorticks_on()
    plt.tick_params(axis='x', which='both', bottom=True, top=True, direction='in')
    plt.tick_params(axis='y', which='both', left=True, right=True, direction='in')
    plt.annotate(f'{celsius}°C', xy=(co2_values[-1], c[-1]), xytext=(-5, 5),
                 textcoords='offset points', ha='right', va='bottom')

plt.subplot(122)
for celsius in [0, 10, 20, 30]:
    c = []
    humidity_values = np.linspace(0, 100, 100)
    for h in humidity_values:
        c.append(c_air(celsius, h, 101325, 0.04))
    plt.plot(humidity_values, c, label=f'{celsius}°C', color='black')
    plt.ylabel('Speed of sound (m/s)')
    plt.xlabel('Humidity (%)')
    plt.ylim(330, 355)
    plt.xlim(0, 100)
    plt.minorticks_on()
    plt.tick_params(axis='x', which='both', bottom=True, top=True, direction='in')
    plt.tick_params(axis='y', which='both', left=True, right=True, direction='in')
    plt.annotate(f'{celsius}°C', xy=(humidity_values[-1], c[-1]), xytext=(-5, 5),
                 textcoords='offset points', ha='right', va='bottom')

plt.tight_layout()
plt.show()