//! Exclusive numeric-loopback socket binding shared with the owning launcher.
#![allow(
    clippy::redundant_pub_crate,
    reason = "Explicit parent scope keeps these implementation helpers private."
)]
use std::{
    io,
    net::{Ipv4Addr, SocketAddrV4},
};
use tokio::net::TcpSocket;

pub(super) fn bound_loopback_socket(port: u16) -> io::Result<TcpSocket> {
    let socket = TcpSocket::new_v4()?;
    bonsai_windows_process::require_exclusive_address(&socket)?;
    socket.bind(SocketAddrV4::new(Ipv4Addr::LOCALHOST, port).into())?;
    Ok(socket)
}
