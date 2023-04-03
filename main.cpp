#include "interface.hpp"

using namespace std;
using namespace cv;

int main(int argc, char *argv[])
{
	char default_left_filename[] = "tsukuba/scene1.row3.col3.ppm";
	char default_right_filename[] = "tsukuba/scene1.row3.col5.ppm";
	char *left_filename = default_left_filename;
	char *right_filename = default_right_filename;
	char *extrinsics_filename = NULL;
	char *intrinsics_filename = NULL;

	GtkBuilder *builder;
	GError *error = NULL;
	ChData *data;

	/* Parse arguments to find left and right filenames */
	// TODO: we should use some library to parse the command line arguments if we
	// are going to use lots of them.
	for (int i = 1; i < argc; i++)
	{
		if (strcmp(argv[i], "-left") == 0)
		{
			i++;
			left_filename = argv[i];
		}
		else if (strcmp(argv[i], "-right") == 0)
		{
			i++;
			right_filename = argv[i];
		}
		else if (strcmp(argv[i], "-extrinsics") == 0)
		{
			i++;
			extrinsics_filename = argv[i];
		}
		else if (strcmp(argv[i], "-intrinsics") == 0)
		{
			i++;
			intrinsics_filename = argv[i];
		}
	}

	Mat left_image = imread(left_filename, 1);

	if (left_image.empty())
	{
		printf("Could not read left image %s.\n", left_filename);
		exit(1);
	}

	Mat right_image = imread(right_filename, 1);

	if (right_image.empty())
	{
		printf("Could not read right image %s.\n", right_filename);
		exit(1);
	}

	if (left_image.size() != right_image.size())
	{
		printf("Left and right images have different sizes.\n");
		exit(1);
	}

	Mat gray_left, gray_right;
	cvtColor(left_image, gray_left, COLOR_BGR2GRAY);
	cvtColor(right_image, gray_right, COLOR_BGR2GRAY);

	/* Create data */
	data = new ChData();

	if (intrinsics_filename != NULL && extrinsics_filename != NULL)
	{
		FileStorage intrinsicsFs(intrinsics_filename, FileStorage::READ);

		if (!intrinsicsFs.isOpened())
		{
			printf("Could not open intrinsic parameters file %s.\n", intrinsics_filename);
			exit(1);
		}

		Mat m1, d1, m2, d2;
		intrinsicsFs["M1"] >> m1;
		intrinsicsFs["D1"] >> d1;
		intrinsicsFs["M2"] >> m2;
		intrinsicsFs["D2"] >> d2;

		FileStorage extrinsicsFs(extrinsics_filename, FileStorage::READ);

		if (!extrinsicsFs.isOpened())
		{
			printf("Could not open extrinsic parameters file %s.\n", extrinsics_filename);
			exit(1);
		}

		printf("Using provided calibration files to undistort and rectify images.\n");

		Mat r, t;
		extrinsicsFs["R"] >> r;
		extrinsicsFs["T"] >> t;

		Mat r1, p1, r2, p2, q;
		data->roi1 = new Rect();
		data->roi2 = new Rect();
		stereoRectify(m1, d1, m2, d2, left_image.size(), r, t, r1, r2, p1, p2, q, CALIB_ZERO_DISPARITY, -1, left_image.size(), data->roi1, data->roi2);

		Mat map11, map12, map21, map22;
		initUndistortRectifyMap(m1, d1, r1, p1, left_image.size(), CV_16SC2, map11, map12);
		initUndistortRectifyMap(m2, d2, r2, p2, right_image.size(), CV_16SC2, map21, map22);

		Mat remapped_left, remapped_right;
		remap(gray_left, remapped_left, map11, map12, INTER_LINEAR);
		remap(gray_right, remapped_right, map21, map22, INTER_LINEAR);

		data->cv_image_left = remapped_left;
		data->cv_image_right = remapped_right;

		Mat color_remapped_left, color_remapped_right;
		remap(left_image, color_remapped_left, map11, map12, INTER_LINEAR);
		remap(right_image, color_remapped_right, map21, map22, INTER_LINEAR);
		left_image = color_remapped_left;
		right_image = color_remapped_right;
	}
	else
	{
		data->cv_image_left = gray_left;
		data->cv_image_right = gray_right;
	}

	/* Init GTK+ */
	gtk_init(&argc, &argv);

	/* Create new GtkBuilder object */
	builder = gtk_builder_new();

	if (!gtk_builder_add_from_file(builder, "StereoTuner.glade", &error))
	{
		g_warning("%s", error->message);
		g_free(error);
		return (1);
	}

	/* Get main window pointer from UI */
	data->main_window = GTK_WIDGET(gtk_builder_get_object(builder, "window1"));
	data->image_left = GTK_IMAGE(gtk_builder_get_object(builder, "image_left"));
	data->image_right = GTK_IMAGE(
		gtk_builder_get_object(builder, "image_right"));
	data->image_depth = GTK_IMAGE(
		gtk_builder_get_object(builder, "image_disparity"));

	data->sc_block_size = GTK_WIDGET(gtk_builder_get_object(builder, "sc_block_size"));
	data->sc_min_disparity = GTK_WIDGET(gtk_builder_get_object(builder, "sc_min_disparity"));
	data->sc_num_disparities = GTK_WIDGET(gtk_builder_get_object(builder, "sc_num_disparities"));
	data->sc_disp_max_diff = GTK_WIDGET(gtk_builder_get_object(builder, "sc_disp_max_diff"));
	data->sc_speckle_range = GTK_WIDGET(gtk_builder_get_object(builder, "sc_speckle_range"));
	data->sc_speckle_window_size = GTK_WIDGET(gtk_builder_get_object(builder, "sc_speckle_window_size"));
	data->sc_p1 = GTK_WIDGET(gtk_builder_get_object(builder, "sc_p1"));
	data->sc_p2 = GTK_WIDGET(gtk_builder_get_object(builder, "sc_p2"));
	data->sc_pre_filter_cap = GTK_WIDGET(gtk_builder_get_object(builder, "sc_pre_filter_cap"));
	data->sc_pre_filter_size = GTK_WIDGET(gtk_builder_get_object(builder, "sc_pre_filter_size"));
	data->sc_uniqueness_ratio = GTK_WIDGET(gtk_builder_get_object(builder, "sc_uniqueness_ratio"));
	data->sc_texture_threshold = GTK_WIDGET(gtk_builder_get_object(builder, "sc_texture_threshold"));
	data->rb_pre_filter_normalized = GTK_WIDGET(gtk_builder_get_object(builder, "rb_pre_filter_normalized"));
	data->rb_pre_filter_xsobel = GTK_WIDGET(gtk_builder_get_object(builder, "rb_pre_filter_xsobel"));
	data->chk_full_dp = GTK_WIDGET(gtk_builder_get_object(builder, "chk_full_dp"));
	data->status_bar = GTK_WIDGET(gtk_builder_get_object(builder, "status_bar"));
	data->pixel_bar = GTK_WIDGET(gtk_builder_get_object(builder, "pixel_bar"));
	data->img_width_bar = GTK_WIDGET(gtk_builder_get_object(builder, "img_width_bar"));
	data->image_disparity_container = GTK_WIDGET(gtk_builder_get_object(builder, "image_disparity_container"));
	data->rb_bm = GTK_WIDGET(gtk_builder_get_object(builder, "algo_sbm"));
	data->rb_sgbm = GTK_WIDGET(gtk_builder_get_object(builder, "algo_ssgbm"));
	data->adj_block_size = GTK_ADJUSTMENT(gtk_builder_get_object(builder, "adj_block_size"));
	data->adj_min_disparity = GTK_ADJUSTMENT(gtk_builder_get_object(builder, "adj_min_disparity"));
	data->adj_num_disparities = GTK_ADJUSTMENT(gtk_builder_get_object(builder, "adj_num_disparities"));
	data->adj_disp_max_diff = GTK_ADJUSTMENT(gtk_builder_get_object(builder, "adj_disp_max_diff"));
	data->adj_speckle_range = GTK_ADJUSTMENT(gtk_builder_get_object(builder, "adj_speckle_range"));
	data->adj_speckle_window_size = GTK_ADJUSTMENT(gtk_builder_get_object(builder, "adj_speckle_window_size"));
	data->adj_p1 = GTK_ADJUSTMENT(gtk_builder_get_object(builder, "adj_p1"));
	data->adj_p2 = GTK_ADJUSTMENT(gtk_builder_get_object(builder, "adj_p2"));
	data->adj_pre_filter_cap = GTK_ADJUSTMENT(gtk_builder_get_object(builder, "adj_pre_filter_cap"));
	data->adj_pre_filter_size = GTK_ADJUSTMENT(gtk_builder_get_object(builder, "adj_pre_filter_size"));
	data->adj_uniqueness_ratio = GTK_ADJUSTMENT(gtk_builder_get_object(builder, "adj_uniqueness_ratio"));
	data->adj_texture_threshold = GTK_ADJUSTMENT(gtk_builder_get_object(builder, "adj_texture_threshold"));
	data->baseline_value = GTK_ENTRY(gtk_builder_get_object(builder, "baseline_value"));
	data->sensor_width_value = GTK_ENTRY(gtk_builder_get_object(builder, "sensor_width_value"));
	data->focallength_value = GTK_ENTRY(gtk_builder_get_object(builder, "focallength_value"));
	data->status_bar_context = gtk_statusbar_get_context_id(GTK_STATUSBAR(data->status_bar), "Statusbar context");
	data->pixel_bar_context = gtk_statusbar_get_context_id(GTK_STATUSBAR(data->pixel_bar), "Pixelbar context");
	data->img_width_bar_context = gtk_statusbar_get_context_id(GTK_STATUSBAR(data->img_width_bar), "img_width_bar context");
	data->pix_rabiobutton = GTK_WIDGET(gtk_builder_get_object(builder, "pix_rabiobutton"));
	data->mm_rabiobutton = GTK_WIDGET(gtk_builder_get_object(builder, "mm_rabiobutton"));
	// Put images in place:
	// gtk_image_set_from_file(data->image_left, left_filename);
	// gtk_image_set_from_file(data->image_right, right_filename);

	Mat leftRGB, rightRGB;
	cvtColor(left_image, leftRGB,
			 COLOR_BGR2RGB);
	GdkPixbuf *pixbuf = gdk_pixbuf_new_from_data(
		(guchar *)leftRGB.data, GDK_COLORSPACE_RGB, false,
		8, leftRGB.cols,
		leftRGB.rows, leftRGB.step,
		NULL, NULL);
	gtk_image_set_from_pixbuf(data->image_left, pixbuf);

	cvtColor(right_image, rightRGB,
			 COLOR_BGR2RGB);
	pixbuf = gdk_pixbuf_new_from_data(
		(guchar *)rightRGB.data, GDK_COLORSPACE_RGB, false,
		8, rightRGB.cols,
		rightRGB.rows, rightRGB.step,
		NULL, NULL);
	gtk_image_set_from_pixbuf(data->image_right, pixbuf);

	update_matcher(data);

	/* Connect signals */
	gtk_builder_connect_signals(builder, data);

	/* Destroy builder, since we don't need it anymore */
	g_object_unref(G_OBJECT(builder));

	/* Show window. All other widgets are automatically shown by GtkBuilder */
	gtk_widget_show(data->main_window);

	/* Start main loop */
	gtk_main();

	return (0);
}
