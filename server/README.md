# server

GPU renderer, server side. Holds a fixed-size brick pool, a page table and a host brick cache. It ray-casts with clip planes and a camera-aligned cut, fetches the bricks that the rays request (cut face first), encodes frames as JPEG and streams them to the browser.
