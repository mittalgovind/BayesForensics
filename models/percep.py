import tensorflow as tf
from models import params


def get_vgg():
    vgg = tf.keras.applications.vgg19.VGG19(include_top=False, weights="imagenet")
    vgg.trainable = False
    style_outputs = [vgg.get_layer(name).output for name in params.style_layers]
    content_outputs = [vgg.get_layer(name).output for name in params.content_layers]
    model_outputs = style_outputs + content_outputs
    return tf.keras.models.Model(vgg.input, model_outputs)


def get_content_loss(content, target):
    return tf.reduce_mean(tf.square(content - target))


def get_loss(img0_feats, img1_feats):
    content_score = 0
    weight_per_content_layer = 1.0 / float(params.num_content_layers)
    for init_img_content_layer, content_img_content_layer in zip(
        img0_feats, img1_feats
    ):
        content_score += weight_per_content_layer * get_content_loss(
            init_img_content_layer, content_img_content_layer
        )
    return content_score


def percep_dist(vgg_model, imgbatch0, imgbatch1):
    # vgg_model = get_vgg()
    # for layer in vgg_model.layers:
    #    layer.trainable = False

    imgbatch0 = tf.keras.applications.vgg19.preprocess_input(imgbatch0)
    imgbatch1 = tf.keras.applications.vgg19.preprocess_input(imgbatch1)

    batch0_vggfeats = vgg_model(imgbatch0)
    batch1_vggfeats = vgg_model(imgbatch1)

    dist = 0

    for each_image_idx in range(len(batch0_vggfeats)):
        img0_feats = [
            content_layer[0]
            for content_layer in batch0_vggfeats[each_image_idx][
                params.num_style_layers :
            ]
        ]
        img1_feats = [
            content_layer[0]
            for content_layer in batch1_vggfeats[each_image_idx][
                params.num_style_layers :
            ]
        ]
        loss = get_loss(img0_feats, img1_feats)
        dist = dist + loss

    return dist
