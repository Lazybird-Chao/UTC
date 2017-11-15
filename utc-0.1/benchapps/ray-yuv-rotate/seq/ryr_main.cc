/*
 * ryr_main.cc
 *
 *  Created on: Nov 13, 2017
 *      Author: Chao
 */


#include "../../common/helper_getopt.h"
#include "../../common/helper_timer.h"
#include "../../common/helper_printtime.h"

#include "common.h"
#include <string>
#include <fstream>
#include <iostream>
#include <cstdlib>
#include <cstdio>
#include <cmath>
#include <cstring>


/*
 * constant variables
 */
#define MAX_LIGHTS		16				/* maximum number of lights */
#define RAY_MAG			1000.0			/* trace rays of this magnitude */
#define MAX_RAY_DEPTH	5				/* raytrace recursion limit */
#define FOV				0.78539816		/* field of view in rads (pi/4) */
#define HALF_FOV		(FOV * 0.5)
#define ERR_MARGIN		1e-6			/* an arbitrary error margin to avoid surface acne */

/*
 * bit-shift ammount for packing each color into a 32bit uint
 */
#ifdef LITTLE_ENDIAN
#define RSHIFT	16
#define BSHIFT	0
#else	/* big endian */
#define RSHIFT	0
#define BSHIFT	16
#endif	/* endianess */
#define GSHIFT	8	/* this is the same in both byte orders */

/* some helpful macros... */
#define SQ(x)		((x) * (x))
#define MAX(a, b)	((a) > (b) ? (a) : (b))
#define MIN(a, b)	((a) < (b) ? (a) : (b))
#define DOT(a, b)	((a).x * (b).x + (a).y * (b).y + (a).z * (b).z)
#define NORMALIZE(a)  do {\
	FTYPE len = sqrt(DOT(a, a));\
	(a).x /= len; (a).y /= len; (a).z /= len;\
} while(0);


/*
 * global variables
 */
#define NRAN	1024
#define MASK	(NRAN - 1)
vec3_t urand[NRAN];
int irand[NRAN];

int xres = 800;
int yres = 600;
int rays_per_pixel = 1;
FTYPE aspect = 1.333333;
sphere_t *obj_list;
vec3_t lights[MAX_LIGHTS];
int lnum = 0;
camera_t cam;
sphere2_t *obj_array; // used to replace the obj_list
int obj_count;


int main(int argc, char **argv){
	bool printTime = false;
	char* infile_path = NULL;
	//char* outfile_path = NULL;

	int loop = 100;

	/* Parse command line options */
	int     opt;
	extern char   *optarg;
	extern int     optind;
	while ( (opt=getopt(argc,argv,"w:h:i:l:v"))!= EOF) {
		switch (opt) {
			case 'v': printTime = true;
					  break;
			case 'i': infile_path=optarg;
					  break;
			case 'w': xres = atoi(optarg);
					  break;
			case 'h': yres = atoi(optarg);
					  break;
			case 'l': loop = atoi(optarg);
					  break;
			case ':':
				std::cerr<<"Option -"<<(char)optopt<<" requires an operand\n"<<std::endl;
				break;
			case '?':
				std::cerr<<"Unrecognized option: -"<<(char)optopt<<std::endl;
				break;
			default:
			    break;
		}
	}

	FILE *infile = nullptr;
	if((infile = fopen((const char*)infile_path, "rb"))==nullptr){
		std::cerr<<"Error, cannot open scene file."<<std::endl;
		return 1;
	}

	/*
	 * read the input scene file
	 */
	if(infile ==nullptr){
		std::cerr<<"Need a input scene file."<<std::endl;
		exit(1);
	}

	load_scene(infile);

	/* initialize the random number tables for the jitter */
	for(int i=0; i<NRAN; i++) urand[i].x = (double)rand() / RAND_MAX - 0.5;
	for(int i=0; i<NRAN; i++) urand[i].y = (double)rand() / RAND_MAX - 0.5;
	for(int i=0; i<NRAN; i++) irand[i] = (int)(NRAN * ((double)rand() / RAND_MAX));

	/*
	 *  the main processing loop
	 */
	uint32_t *pixels_array = (uint32_t*)malloc(xres * yres * sizeof(int)*loop);
	uint8_t *r_array = new uint8_t[xres * yres*loop];
	uint8_t *g_array = new uint8_t[xres * yres*loop];
	uint8_t *b_array = new uint8_t[xres * yres*loop];

	double runtime[5] = {0,0,0,0,0};

	int iter = 0;
	while(iter < loop){

		uint32_t *pixels = pixels_array + (xres*yres*iter);
		/*
		 * call raytrace to generate a picutre
		 */
		double t1, t2;
		t1 = getTime();
		for(int i=0; i<yres; i++){
			render_scanline(xres,
							yres,
							i,
							(uint32_t*)(pixels + i*xres),
							rays_per_pixel);

		}
		t2 = getTime();
		runtime[1] += t2-t1;
		//std::cout<<"finish ray trace...\n";

		/*
		 * do coloer convertion
		 */
		t1 = getTime();
		uint8_t *r = r_array + xres*yres*iter;
		uint8_t *g = g_array + xres*yres*iter;
		uint8_t *b = b_array + xres*yres*iter;
		for(int i =0; i < xres*yres; i++){
			r[i] = (pixels[i]>>RSHIFT) & 0xff;
			g[i] = (pixels[i]>>GSHIFT) & 0xff;
			b[i] = (pixels[i]>>BSHIFT) & 0xff;
		}
		uint8_t *y = nullptr;
		uint8_t *u = nullptr;
		uint8_t *v = nullptr;
		yuv(5, xres, yres, r, g, b, &y, &u, &v);
		t2 = getTime();
		runtime[2] += t2 - t1;
		//std::cout<<"finish yuv...\n";

		/*
		 * do rotation
		 */
		t1 = getTime();
		Image *finalImage1 = nullptr;
		Image *finalImage2 = nullptr;
		Image *finalImage3 = nullptr;
		rotate(xres, yres, y, &finalImage1);
		rotate(xres, yres, u, &finalImage2);
		rotate(xres, yres, v, &finalImage3);
		t2 = getTime();
		runtime[3] += t2 - t1;
		//std::cout<<"finish rotate...\n";

		/*
		 * out put data
		 */
		t1 = getTime();
		output(iter*3+0, finalImage1);
		output(iter*3+1, finalImage2);
		output(iter*3+2, finalImage3);
		t2 = getTime();
		runtime[4] += t2 - t1;
		std::cout<<"finish round "<<iter<<std::endl;

		iter++;
	}

	std::cout<<"Test complete !!!"<<std::endl;
	runtime[0] = runtime[1]+runtime[2]+runtime[3]+runtime[4];
	if(printTime){
		std::cout<<"Output image: "<<xres<<" X "<<yres<<std::endl;
		std::cout<<"Total time: "<<std::fixed<<std::setprecision(4)<<runtime[0]<<"(s)"<<std::endl;
		std::cout<<"raytrace time: "<<std::fixed<<std::setprecision(4)<<runtime[1]<<"(s)"<<std::endl;
		std::cout<<"yuv time: "<<std::fixed<<std::setprecision(4)<<runtime[2]<<"(s)"<<std::endl;
		std::cout<<"rotate time: "<<std::fixed<<std::setprecision(4)<<runtime[3]<<"(s)"<<std::endl;
		std::cout<<"output time: "<<std::fixed<<std::setprecision(4)<<runtime[4]<<"(s)"<<std::endl;
	}
	for(int i = 0; i<5; i++)
		runtime[i] *= 1000;
	print_time(5, runtime);


}


void output(int iter, Image *dstImg){
	std::string outfile_path = "./outfile/";
	outfile_path += std::to_string(iter);
	outfile_path += ".ppm";
	std::fstream out;
	out.open(outfile_path.c_str(), std::fstream::out);
	out<<"P6\n";
	out << dstImg->getWidth() << " " << dstImg->getHeight() << "\n" << dstImg->getMaxcolor() << "\n";
	for(int i = 0; i < dstImg->getHeight(); i++) {
		for(int j = 0; j < dstImg->getWidth(); j++) {
			Pixel p = dstImg->getPixelAt(j, i);
			out.put(p.r);
			out.put(p.g);
			out.put(p.b);
		}
	}
	out.close();
}


/* render a frame of xsz/ysz dimensions into the provided framebuffer */
void render_scanline(int xsz, int ysz, int sl, uint32_t *fb, int samples){
	int i, s;
	FTYPE rcp_samples = 1.0 / (FTYPE)samples;

	for(i=0; i<xsz; i++) {
		double r, g, b;
		r = g = b = 0.0;

		for(s=0; s<samples; s++) {
			vec3_t col = trace(get_primary_ray(i, sl, s), 0);
			r += col.x;
			g += col.y;
			b += col.z;
		}

		r = r * rcp_samples;
		g = g * rcp_samples;
		b = b * rcp_samples;

		fb[i] = ((uint32_t)(MIN(r, 1.0) * 255.0) & 0xff) << RSHIFT |
							((uint32_t)(MIN(g, 1.0) * 255.0) & 0xff) << GSHIFT |
							((uint32_t)(MIN(b, 1.0) * 255.0) & 0xff) << BSHIFT;
	}
}

/*
 * trace a ray throught the scene recursively (the recursion happens through
 * shade() to calculate reflection rays if necessary).
 */
vec3_t trace(ray_t ray, int depth) {
	vec3_t col;
	spoint_t sp, nearest_sp;
	sphere2_t *nearest_obj=nullptr;
	int obj_iter = 0;

	if(depth >= MAX_RAY_DEPTH){
		col.x = col.y = col.z = 0.0;
		return col;
	}

	/* find the nearest intersection ... */
	while(obj_iter < obj_count) {
		if(ray_sphere(&obj_array[obj_iter], ray, &sp)) {
			if(!nearest_obj || sp.dist < nearest_sp.dist) {
				nearest_obj = &obj_array[obj_iter];
				nearest_sp = sp;
			}
		}
		obj_iter++;
	}

	/* and perform shading calculations as needed by calling shade() */
	if(nearest_obj) {
		col = shade(nearest_obj, &nearest_sp, depth);
	} else {
		col.x = col.y = col.z = 0.0;
	}

	return col;
}


/*
 *  Calculates direct illumination with the phong reflectance model.
 * Also handles reflections by calling trace again, if necessary.
 */
vec3_t shade( sphere2_t *obj, spoint_t *sp, int depth) {
	int i;
	vec3_t col = {0,0,0};

	/* for all lights ... */
		for(i=0; i<lnum; i++) {
			FTYPE ispec, idiff;
			vec3_t ldir;
			ray_t shadow_ray;
			int obj_iter = 0;//struct sphere *iter = obj_list->next;
			int in_shadow = 0;

			ldir.x = lights[i].x - sp->pos.x;
			ldir.y = lights[i].y - sp->pos.y;
			ldir.z = lights[i].z - sp->pos.z;

			shadow_ray.orig = sp->pos;
			shadow_ray.dir = ldir;

			/* shoot shadow rays to determine if we have a line of sight with the light */
			while(obj_iter < obj_count) {
				if(ray_sphere(&obj_array[obj_iter], shadow_ray, 0)) {
					in_shadow = 1;
					break;
				}
				obj_iter++;
			}

			/* and if we're not in shadow, calculate direct illumination with the phong model. */
			if(!in_shadow) {
				NORMALIZE(ldir);

				idiff = MAX(DOT(sp->normal, ldir), 0.0);
				ispec = obj->mat.spow > 0.0 ? pow(MAX(DOT(sp->vref, ldir), 0.0), obj->mat.spow) : 0.0;

				col.x += idiff * obj->mat.col.x + ispec;
				col.y += idiff * obj->mat.col.y + ispec;
				col.z += idiff * obj->mat.col.z + ispec;
			}
		}

		/* Also, if the object is reflective, spawn a reflection ray, and call trace()
		 * to calculate the light arriving from the mirror direction.
		 */
		if(obj->mat.refl > 0.0) {
			ray_t ray;
			vec3_t rcol;

			ray.orig = sp->pos;
			ray.dir = sp->vref;
			ray.dir.x *= RAY_MAG;
			ray.dir.y *= RAY_MAG;
			ray.dir.z *= RAY_MAG;

			rcol = trace(ray, depth + 1);
			col.x += rcol.x * obj->mat.refl;
			col.y += rcol.y * obj->mat.refl;
			col.z += rcol.z * obj->mat.refl;
		}

		return col;
}

/*
 * Calculate ray-sphere intersection, and return {1, 0} to signify hit or no hit.
 * Also the surface point parameters like position, normal, etc are returned through
 * the sp pointer if it is not NULL.
 */
int ray_sphere(const sphere2_t *sph, ray_t ray, spoint_t *sp) {
	FTYPE a, b, c, d, sqrt_d, t1, t2;

	a = SQ(ray.dir.x) + SQ(ray.dir.y) + SQ(ray.dir.z);
	b = 2.0 * ray.dir.x * (ray.orig.x - sph->pos.x) +
				2.0 * ray.dir.y * (ray.orig.y - sph->pos.y) +
				2.0 * ray.dir.z * (ray.orig.z - sph->pos.z);
	c = SQ(sph->pos.x) + SQ(sph->pos.y) + SQ(sph->pos.z) +
				SQ(ray.orig.x) + SQ(ray.orig.y) + SQ(ray.orig.z) +
				2.0 * (-sph->pos.x * ray.orig.x - sph->pos.y * ray.orig.y - sph->pos.z * ray.orig.z) - SQ(sph->rad);

	if((d = SQ(b) - 4.0 * a * c) < 0.0) return 0;

	sqrt_d = sqrt(d);
	t1 = (-b + sqrt_d) / (2.0 * a);
	t2 = (-b - sqrt_d) / (2.0 * a);

	if((t1 < ERR_MARGIN && t2 < ERR_MARGIN) || (t1 > 1.0 && t2 > 1.0)) return 0;

	if(sp) {
		if(t1 < ERR_MARGIN) t1 = t2;
		if(t2 < ERR_MARGIN) t2 = t1;
		sp->dist = t1 < t2 ? t1 : t2;

		sp->pos.x = ray.orig.x + ray.dir.x * sp->dist;
		sp->pos.y = ray.orig.y + ray.dir.y * sp->dist;
		sp->pos.z = ray.orig.z + ray.dir.z * sp->dist;

		sp->normal.x = (sp->pos.x - sph->pos.x) / sph->rad;
		sp->normal.y = (sp->pos.y - sph->pos.y) / sph->rad;
		sp->normal.z = (sp->pos.z - sph->pos.z) / sph->rad;

		sp->vref = reflect(ray.dir, sp->normal);
		NORMALIZE(sp->vref);
	}
	return 1;

}


/* determine the primary ray corresponding to the specified pixel (x, y) */
ray_t get_primary_ray(int x, int y, int sample) {
	ray_t ray;
	FTYPE m[3][3];
	vec3_t i, j = {0, 1, 0}, k, dir, orig, foo;

	k.x = cam.targ.x - cam.pos.x;
	k.y = cam.targ.y - cam.pos.y;
	k.z = cam.targ.z - cam.pos.z;
	NORMALIZE(k);

	i = cross_product(j, k);
	j = cross_product(k, i);
	m[0][0] = i.x; m[0][1] = j.x; m[0][2] = k.x;
	m[1][0] = i.y; m[1][1] = j.y; m[1][2] = k.y;
	m[2][0] = i.z; m[2][1] = j.z; m[2][2] = k.z;

	ray.orig.x = ray.orig.y = ray.orig.z = 0.0;
	ray.dir = get_sample_pos(x, y, sample);
	ray.dir.z = 1.0 / HALF_FOV;
	ray.dir.x *= RAY_MAG;
	ray.dir.y *= RAY_MAG;
	ray.dir.z *= RAY_MAG;

	dir.x = ray.dir.x + ray.orig.x;
	dir.y = ray.dir.y + ray.orig.y;
	dir.z = ray.dir.z + ray.orig.z;
	foo.x = dir.x * m[0][0] + dir.y * m[0][1] + dir.z * m[0][2];
	foo.y = dir.x * m[1][0] + dir.y * m[1][1] + dir.z * m[1][2];
	foo.z = dir.x * m[2][0] + dir.y * m[2][1] + dir.z * m[2][2];

	orig.x = ray.orig.x * m[0][0] + ray.orig.y * m[0][1] + ray.orig.z * m[0][2] + cam.pos.x;
	orig.y = ray.orig.x * m[1][0] + ray.orig.y * m[1][1] + ray.orig.z * m[1][2] + cam.pos.y;
	orig.z = ray.orig.x * m[2][0] + ray.orig.y * m[2][1] + ray.orig.z * m[2][2] + cam.pos.z;

	ray.orig = orig;
	ray.dir.x = foo.x + orig.x;
	ray.dir.y = foo.y + orig.y;
	ray.dir.z = foo.z + orig.z;

	return ray;
}


vec3_t get_sample_pos(int x, int y, int sample) {
	vec3_t pt;
	static FTYPE sf = 0.0;

	if(sf == 0.0) {
		sf = 1.5 / (FTYPE)xres;
	}

	pt.x = ((FTYPE)x / (FTYPE)xres) - 0.5;
	pt.y = -(((FTYPE)y / (FTYPE)yres) - 0.65) / aspect;

	if(sample) {
		vec3_t jt = jitter(x, y, sample);
		pt.x += jt.x * sf;
		pt.y += jt.y * sf / aspect;
	}
	return pt;
}

/* jitter function taken from Graphics Gems I. */
inline vec3_t jitter(int x, int y, int s) {
	vec3_t pt;
	pt.x = urand[(x + (y << 2) + irand[(x + s) & MASK]) & MASK].x;
	pt.y = urand[(y + (x << 2) + irand[(y + s) & MASK]) & MASK].y;
	return pt;
}


/*
 * some help routine
 */
inline vec3_t reflect(vec3_t &v, vec3_t &n) {
	vec3_t res;
	FTYPE dot = v.x * n.x + v.y * n.y + v.z * n.z;
	res.x = -(2.0 * dot * n.x - v.x);
	res.y = -(2.0 * dot * n.y - v.y);
	res.z = -(2.0 * dot * n.z - v.z);
	return res;
}

inline vec3_t cross_product(vec3_t &v1, vec3_t &v2) {
	vec3_t res;
	res.x = v1.y * v2.z - v1.z * v2.y;
	res.y = v1.z * v2.x - v1.x * v2.z;
	res.z = v1.x * v2.y - v1.y * v2.x;
	return res;
}


/* Load the scene from an extremely simple scene description file */
#define DELIM	" \t\n"
void load_scene(FILE *fp) {
	char line[256], *ptr, type;

	obj_list = (sphere_t*)malloc(sizeof(struct sphere));
	obj_list->next = nullptr;
	obj_count=0;

	while((ptr = fgets(line, 256, fp))) {
		int i;
		vec3_t pos, col;
		FTYPE rad, spow, refl;

		while(*ptr == ' ' || *ptr == '\t') ptr++;
		if(*ptr == '#' || *ptr == '\n') continue;

		if(!(ptr = strtok(line, DELIM))) continue;
		type = *ptr;

		for(i=0; i<3; i++) {
			if(!(ptr = strtok(0, DELIM))) break;
			*((FTYPE*)&pos.x + i) = (FTYPE)atof(ptr);
		}

		if(type == 'l') {
			lights[lnum++] = pos;
			continue;
		}

		if(!(ptr = strtok(0, DELIM))) continue;
		rad = atof(ptr);

		for(i=0; i<3; i++) {
			if(!(ptr = strtok(0, DELIM))) break;
			*((FTYPE*)&col.x + i) = (FTYPE)atof(ptr);
		}

		if(type == 'c') {
			cam.pos = pos;
			cam.targ = col;
			cam.fov = rad;
			continue;
		}

		if(!(ptr = strtok(0, DELIM))) continue;
		spow = (FTYPE)atof(ptr);

		if(!(ptr = strtok(0, DELIM))) continue;
		refl = (FTYPE)atof(ptr);

		if(type == 's') {
			obj_count++;
			struct sphere *sph = (sphere_t*)malloc(sizeof *sph);
			sph->next = obj_list->next;
			obj_list->next = sph;

			sph->pos = pos;
			sph->rad = rad;
			sph->mat.col = col;
			sph->mat.spow = spow;
			sph->mat.refl = refl;
		} else {
			fprintf(stderr, "unknown type: %c\n", type);
		}
	}

	/*
	 * change the sphere linked list to an array
	 */
	obj_array = (sphere2_t*)malloc(obj_count * sizeof(sphere2_t));
	sphere_t *p1 = obj_list->next;
	sphere_t *p2 = p1;
	int i=0;
	while(p1!=nullptr){
		obj_array[i].pos = p1->pos;
		obj_array[i].rad = p1->rad;
		obj_array[i].mat.col = p1->mat.col;
		obj_array[i].mat.spow = p1->mat.spow;
		obj_array[i].mat.refl = p1->mat.refl;
		p2 = p1;
		p1 = p1->next;
		free(p2);
		i++;
	}
	obj_list->next = nullptr;
	free(obj_list);
}
