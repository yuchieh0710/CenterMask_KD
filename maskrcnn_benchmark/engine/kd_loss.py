import torch.nn.functional as F
import torch
from torch import nn


def pixel_wise_loss(s_feature, t_feature, mode):
    class_num = s_feature[0].shape[1]
    loss_pixel_wise = 0
    for i in range(len(s_feature)):
        s = F.log_softmax(s_feature[i].permute(0, 2, 3, 1).contiguous().view(-1, class_num))
        t = F.softmax(t_feature[i].permute(0, 2, 3, 1).contiguous().view(-1, class_num))
        if mode == 'KL':
            loss = nn.KLDivLoss(reduction='batchmean')(s, t)
        elif mode == 'cross-entropy':
            loss = (torch.sum(-t * s)) / t_feature[i].shape[2] / t_feature[i].shape[3] / t_feature[i].shape[0]
        loss_pixel_wise += loss
    loss_pixel_wise / len(s_feature)

    return loss_pixel_wise


def new_box_loss(t_loss_reg, s_loss_reg):
    if s_loss_reg > t_loss_reg:
        loss_regression = s_loss_reg + 0.5 * s_loss_reg
    else:
        loss_regression = s_loss_reg
    return loss_regression


def new_center_loss(t_loss_center, s_loss_center):
    if s_loss_center > t_loss_center:
        loss_center = s_loss_center + 0.5 * s_loss_center
    else:
        loss_center = s_loss_center
    return loss_center


def hook_s(module, input, output):
    s_outputs.append(output)


def hook_t(module, input, output):
    t_outputs.append(output)


def hook_ins(module, input, output):
    s_inputs.append(output)


def hook_int(module, input, output):
    t_inputs.append(output)


def featuremap_loss(feature_t, feature_s):
    criterion = nn.MSELoss()
    loss = 0

    for i in range(len(feature_t)):
        loss += criterion(feature_s[i], feature_t[i])

    return loss


def get_correlation(layer1, layer2):
    if layer1.size(2) > layer2.size(2):
        layer1 = F.adaptive_avg_pool2d(layer1, (layer2.size(2), layer2.size(3)))

    layer1 = layer1.view(layer1.size(0), layer1.size(1), -1)
    layer2 = layer2.view(layer2.size(0), layer2.size(1), -1).permute(0, 2, 1)

    correlation = torch.bmm(layer1, layer2) / layer1.size(2)
    return correlation


def correlation_loss(attention_t, attention_s):
    criterion = nn.MSELoss()
    loss = 0
    for i in range(len(attention_t) - 1):
        correlation_s = get_correlation(attention_s[i], attention_s[i + 1])
        correlation_t = get_correlation(attention_t[i], attention_t[i + 1])
        loss += criterion(correlation_s, correlation_t)

    return loss


def get_feature(t_model, model, images, target, correlation):
    global s_outputs
    global t_outputs
    s_outputs = []
    t_outputs = []

    global s_inputs
    global t_inputs
    s_inputs = []
    t_inputs = []

    attention_s = []
    attention_t = []
    s_features_list = [model.backbone.body.stage2.OSA2_1.ese.hsigmoid, model.backbone.body.stage3.OSA3_1.ese.hsigmoid,
                       model.backbone.body.stage4.OSA4_1.ese.hsigmoid, model.backbone.body.stage5.OSA5_1.ese.hsigmoid,
                       model.backbone.fpn.conv]
    s_input_list = [model.backbone.body.stage2.OSA2_1.concat[2], model.backbone.body.stage3.OSA3_1.concat[2],
                    model.backbone.body.stage4.OSA4_1.concat[2], model.backbone.body.stage5.OSA5_1.concat[2]]
    t_features_list = [t_model.backbone.body.stage2.OSA2_1.ese.hsigmoid,
                       t_model.backbone.body.stage3.OSA3_1.ese.hsigmoid,
                       t_model.backbone.body.stage4.OSA4_1.ese.hsigmoid,
                       t_model.backbone.body.stage5.OSA5_1.ese.hsigmoid,
                       t_model.backbone.fpn.top_blocks.p7]
    t_input_list = [t_model.backbone.body.stage2.OSA2_1.concat[2], t_model.backbone.body.stage3.OSA3_1.concat[2],
                    t_model.backbone.body.stage4.OSA4_1.concat[2], t_model.backbone.body.stage5.OSA5_1.concat[2]]
    with torch.no_grad():
        handle_s0 = s_features_list[0].register_forward_hook(hook_s)
        handle_s1 = s_features_list[1].register_forward_hook(hook_s)
        handle_s2 = s_features_list[2].register_forward_hook(hook_s)
        handle_s3 = s_features_list[3].register_forward_hook(hook_s)
        handle_s4 = s_features_list[4].register_forward_hook(hook_s)
        model(images, target)
        handle_s0.remove()
        handle_s1.remove()
        handle_s2.remove()
        handle_s3.remove()
        handle_s4.remove()

        handle_t0 = t_features_list[0].register_forward_hook(hook_t)
        handle_t1 = t_features_list[1].register_forward_hook(hook_t)
        handle_t2 = t_features_list[2].register_forward_hook(hook_t)
        handle_t3 = t_features_list[3].register_forward_hook(hook_t)
        handle_t4 = t_features_list[4].register_forward_hook(hook_t)
        t_model(images, target)
        handle_t0.remove()
        handle_t1.remove()
        handle_t2.remove()
        handle_t3.remove()
        handle_t4.remove()

        handle_s0 = s_input_list[0].register_forward_hook(hook_ins)
        handle_s1 = s_input_list[1].register_forward_hook(hook_ins)
        handle_s2 = s_input_list[2].register_forward_hook(hook_ins)
        handle_s3 = s_input_list[3].register_forward_hook(hook_ins)
        model(images, target)
        handle_s0.remove()
        handle_s1.remove()
        handle_s2.remove()
        handle_s3.remove()

        handle_t0 = t_input_list[0].register_forward_hook(hook_int)
        handle_t1 = t_input_list[1].register_forward_hook(hook_int)
        handle_t2 = t_input_list[2].register_forward_hook(hook_int)
        handle_t3 = t_input_list[3].register_forward_hook(hook_int)
        t_model(images, target)
        handle_t0.remove()
        handle_t1.remove()
        handle_t2.remove()
        handle_t3.remove()

        for i in range(len(s_inputs)):
            attention_s.append(s_inputs[i] * s_outputs[i])
            attention_t.append(t_inputs[i] * t_outputs[i])
        attention_s.append(s_outputs[4])
        attention_t.append(t_outputs[4])

        if correlation:
            loss = correlation_loss(attention_t, attention_s)
        else:
            loss = featuremap_loss(attention_t, attention_s)
        loss_intermediate = loss / len(t_outputs)

    return loss_intermediate

