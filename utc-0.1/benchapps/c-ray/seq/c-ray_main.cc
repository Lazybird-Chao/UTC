/*
 * c-ray_main.cc
 *
 * The simplified sequential ray-tracing program.
 *
 * usage:
 * 		compile with the Makefile
 * 		run as: ./a.out -i scene_file -o img_file -v -w 800 -h 600 -r 1
 * 			-v: print time info
 * 			-i: input scene file path
 * 			-o: output image file path
 * 			-w: output image x resolution
 * 			-h: output image y resolution
 * 			-r: the number of rays per pixel
 *
 *
 * Scene file format:
 *   # sphere (many)
 *   s  x y z  rad   r g b   shininess   reflectivity
 *   # light (many)
 *   l  x y z
 *   # camera (one)
 *   c  x y z  fov   tx ty tz
 *
 *
 */

#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <iomanip>
#include <cmath>
#include <cerrno>


#include "../../common/helper_getopt.h"
#include "../../common/helper_timer.h"

#define FTYPE double


/*
 * required data structure
 */
typedef struct vec3{
	FTYPE x,y,z;
}	vec3_t;

typedef struct ray{
	struct vec3 orig, dir;
}	ray_t;

typedef struct material {
	struct vec3 col;	/* color */
	FTYPE spow;		/* specular power */
	FTYPE refl;		/* reflection intensity */
}	material_t;

typedef struct sphere {
	struct vec3 pos;
	FTYPE rad;
	struct material mat;
	struct sphere *next;
}	sphere_t;

typedef struct spoint {
	struct vec3 pos, normal, vref;	/* position, normal and view reflection */
	FTYPE dist;		/* parametric distance of intersection along the ray */
}	spoint_t;

typedef struct camera {
	struct vec3 pos, targ;
	FTYPE fov;
}	camera_t;

typedef struct sphere2 {
	struct vec3 pos;
	FTYPE rad;
	struct material mat;
}	sphere2_t;

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

/*
 * sub routines declaration
 */
void render_scanline(int xsz, int ysz, int sl, uint32_t *fb, int samples);
vec3_t	trace(ray_t ray, int depth);
vec3_t	shade(sphere2_t *obj, spoint_t *sp, int depth);
inline vec3_t	reflect(vec3_t &v, vec3_t &n);
inline vec3_t	cross_product(vec3_t &v1, vec3_t &v2);
ray_t	get_primary_ray(int x, int y, int sample);
vec3_t	get_sample_pos(int x, int y, int sample);
inline vec3_t	jitter(int x, int y, int s);
int 	ray_sphere(const sphere2_t *sph, ray_t ray, spoint_t *sp);
void 	load_scene(FILE *fp);


int main(int argc, char **argv){
	FILE *infile = nullptr;
	FILE *outfile = nullptr;
	uint32_t *pixels;
	bool printTime = false;
	char* infile_path;
	char* outfile_path;

	/* Parse command line options */
	int     opt;
	extern char   *optarg;
	extern int     optind;
	while ( (opt=getopt(argc,argv,"w:h:r:i:o:v"))!= EOF) {
		switch (opt) {
			case 'v': printTime = true;
					  break;
			case 'i': infile_path=optarg;
					  break;
			case 'o': outfile_path = optarg;
					  break;
			case 'w': xres = atoi(optarg);
					  break;
			case 'h': yres = atoi(optarg);
					  break;
			case 'r': rays_per_pixel = atoi(optarg);
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

	if((infile = fopen((const char*)infile_path, "rb"))==nullptr){
		std::cerr<<"Error, cannot open scene file."<<std::endl;
	}
	if((outfile = fopen((const char*)outfile_path, "wb"))==nullptr){
		std::cerr<<"Error, cannot open output file."<<std::endl;
	}

	if(!(pixels = (uint32_t*)malloc(xres * yres * sizeof *pixels))) {
		perror("pixel buffer allocation failed");
		return EXIT_FAILURE;
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
	 * ray tracing routine
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
	double runtime = t2-t1;

	/*
	 * output the image
	 */
	if(outfile != nullptr){
		fprintf(outfile, "P6\n%d %d\n255\n", xres, yres);
		for(int i=0; i<xres * yres; i++) {
			fputc((pixels[i] >> RSHIFT) & 0xff, outfile);
			fputc((pixels[i] >> GSHIFT) & 0xff, outfile);
			fputc((pixels[i] >> BSHIFT) & 0xff, outfile);
		}
		fflush(outfile);
	}
	if(infile != stdin) fclose(infile);
	if(outfile != stdout) fclose(outfile);
	if(obj_array)
		free(obj_array);
	free(pixels);

	if(printTime){
		std::cout<<"Scene info:"<<std::endl;
		std::cout<<"\tNumber of objects: "<<obj_count<<std::endl;
		std::cout<<"\tNumber of lights: "<<lnum<<std::endl;
		std::cout<<"\tTracing depth: "<<MAX_RAY_DEPTH<<std::endl;
		std::cout<<"Output image: "<<xres<<" X "<<yres<<std::endl;
		std::cout<<"Runtime: "<<std::fixed<<std::setprecision(4)<<runtime<<"(s)"<<std::endl;
	}

	return 0;


}// end main

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















