from matplotlib import pyplot
import numpy as np


def plot_F_over_attack(scores, markers, colors, attack_range, prefix, title):
    pyplot.figure(figsize=(4, 4))
    for i in range(scores.shape[0]):
        method = scores[i]
        pyplot.title(title)
        pyplot.plot(method,  marker=markers[i], color=colors[i])
        pyplot.draw()
        pyplot.legend(legends, loc=4, framealpha=.25)
        ticks = [prefix + str(k) for k in range(1, attack_range)]
        pyplot.xticks(range(0, attack_range-1), ticks)
        pyplot.ylim([0, .6])

    pyplot.savefig('fig6_color/fig6_{}.png'.format(title))
    pyplot.close()


cr_scores = np.array([[0.3791, 0.4331, 0.3156, 0.4917, 0.5657],
                      [0.3743, 0.4247, 0.3039, 0.4893, 0.5633],
                      [0.3698, 0.4159, 0.3138, 0.4865, 0.561]])

ib_scores = np.array([[0.3698, 0.4106, 0.2814, 0.4746, 0.5359],
                      [0.3559, 0.3977, 0.2424, 0.4098, 0.4631],
                      [0.3405, 0.3787, 0.1973, 0.3842, 0.4333]])

na_scores = np.array([[0.092,  0.1806, 0.2649, 0.424,  0.5051],
                      [0.1675, 0.2897, 0.2871, 0.4375, 0.5086],
                      [0.3436, 0.3645, 0.305,  0.4795, 0.5495]])

jc_scores = np.array([[0.1729, 0.3084, 0.1604, 0.297,  0.3711],
                      [0.2533, 0.3375, 0.1842, 0.3557, 0.4179],
                      [0.2841, 0.3464, 0.2057, 0.3663, 0.4381],
                      [0.2967, 0.3527, 0.2159, 0.4191, 0.4908],
                      [0.3109, 0.3635, 0.2288, 0.4073, 0.4897],
                      [0.3209, 0.3703, 0.2438, 0.4281, 0.4991],
                      [0.3463, 0.3886, 0.2837, 0.4408, 0.5225],
                      [0.3566, 0.4116, 0.3003, 0.4633, 0.5395],
                      [0.3287, 0.3973, 0.2823, 0.4393, 0.5115]])

ca_scores = np.array([[0.3816, 0.4346, 0.3056, 0.4939, 0.568 ],
                      [0.3817, 0.4241, 0.3054, 0.4942, 0.5692],
                      [0.3802, 0.425,  0.3165, 0.4942, 0.5743]])

bc_scores = np.array([[0.3732, 0.416,  0.3051, 0.489,  0.5661],
                      [0.3681, 0.4185, 0.3143, 0.4782, 0.5532],
                      [0.3609, 0.3924, 0.3126, 0.4616, 0.5311]])


markers = ["o", "x", "+", "*", "D"]
colors = ["green", "darkviolet", "royalblue", "darkorange", "crimson"]
legends = ['Ryu2010', 'Cozzolino2015', 'Wu2017', 'BusterNet', 'Proposed']


plot_F_over_attack(cr_scores.T, markers, colors, attack_range=4, prefix="CR", title="Color_Reduction(CR)")
plot_F_over_attack(ib_scores.T, markers, colors, attack_range=4, prefix="IB", title="Image_Blurring(IB)")
plot_F_over_attack(na_scores.T, markers, colors, attack_range=4, prefix="NA", title="Noise_Adding(NA)")
plot_F_over_attack(jc_scores.T, markers, colors, attack_range=11, prefix="JC", title="JPEG_Compression(JC)")
plot_F_over_attack(ca_scores.T, markers, colors, attack_range=4, prefix="CA", title="Contrast_Adjustment(CA)")
plot_F_over_attack(bc_scores.T, markers, colors, attack_range=4, prefix="BC", title="Brightness_Change(BC)")


